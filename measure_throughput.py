import os
import sys
import time
import argparse
from pathlib import Path

import torch
from torch.cuda.amp import autocast


def build_argparser():
	p = argparse.ArgumentParser(
		description="Measure throughput (scenes/sec) for S3DIS model with batch scenes of fixed points"
	)
	p.add_argument("--batch-size", type=int, default=16, help="Scenes per batch")
	p.add_argument("--points", type=int, default=15000, help="Points per scene")
	p.add_argument("--partition", type=str, default="!5", help="S3DIS partition mask (e.g., '!5' for training areas)")
	p.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")
	p.add_argument("--warmup-batches", type=int, default=50, help="Batches to warm up before timing")
	p.add_argument("--measure-batches", type=int, default=50, help="Batches to time for throughput")
	p.add_argument("--shuffle", action="store_true", help="Shuffle dataset")
	p.add_argument("--pin-memory", action="store_true", help="Enable pin_memory in DataLoader")
	p.add_argument("--persist", action="store_true", help="Enable persistent_workers in DataLoader")
	p.add_argument("--ckpt", type=str, default="pretrained/best.pt", help="Optional checkpoint to load (relative to S3DIS dir)")
	p.add_argument(
		"--mode",
		choices=["forward", "full"],
		default="forward",
		help="Timing mode: 'forward' measures only model.forward; 'full' includes dataloading and H2D copies",
	)
	return p


def main():
	args = build_argparser().parse_args()

	torch.set_float32_matmul_precision("high")
	torch.backends.cudnn.benchmark = True

	# Ensure we import from S3DIS folder and that its relative data paths resolve
	repo_root = Path(__file__).resolve().parent
	s3dis_dir = repo_root / "S3DIS"
	os.chdir(s3dis_dir)
	sys.path.append(str(s3dis_dir))
	sys.path.append(str(s3dis_dir.parent))  # for utils/

	# Late imports after chdir so config paths are resolved correctly
	from s3dis import S3DIS, s3dis_collate_fn  # type: ignore
	from config import s3dis_args, dela_args, model_type  # type: ignore

	# Choose model implementation based on config
	if model_type == "dela_semseg":
		from delasemseg import DelaSemSeg as ModelClass  # type: ignore
		print("Using model: DelaSemSeg")
	elif model_type in ("dela_semseg_attn", "global_dela", "GDLA-Light", "GDLA-Heavy"):
		from global_dela import DelaSemSeg as ModelClass  # type: ignore
	elif model_type == "dela_semseg_baseline":
		from delasemseg_baseline import DelaSemSeg as ModelClass  # type: ignore
	else:
		raise ValueError(f"Unknown model_type: {model_type}")

	# Enforce point budget per scene by using training mode dataset (for cropping)
	s3dis_args.max_pts = int(args.points)

	dataset = S3DIS(
		s3dis_args,
		partition=args.partition,
		loop=30,
		train=True,
		test=False,
		warmup=False,
	)

	dl = torch.utils.data.DataLoader(
		dataset,
		batch_size=int(args.batch_size),
		shuffle=bool(args.shuffle),
		drop_last=True,
		collate_fn=s3dis_collate_fn,
		num_workers=int(args.num_workers),
		pin_memory=bool(args.pin_memory),
		persistent_workers=bool(args.persist),
	)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if device.type != "cuda":
		print("Warning: CUDA not available; measurements will be on CPU and not representative.")

	model = ModelClass(dela_args).to(device).eval()

	# Load checkpoint if available
	ckpt_path = (s3dis_dir / args.ckpt).resolve()
	if ckpt_path.is_file():
		try:
			import utils.util as util  # type: ignore
			util.load_state(str(ckpt_path), model=model)
			print(f"Loaded checkpoint: {ckpt_path}")
		except Exception as e:
			print(f"Could not load checkpoint {ckpt_path}: {e}")
	else:
		print(f"Checkpoint not found: {ckpt_path} (continuing with randomly initialized weights)")

	# Report model parameter count
	try:
		total_params = sum(p.numel() for p in model.parameters())
		trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
		print(f"Model parameters: {total_params:,} total | {trainable_params:,} trainable ({total_params/1e6:.2f} M)")
	except Exception as e:
		print(f"Could not compute parameter count: {e}")

	# Warmup
	warmup_batches = max(0, int(args.warmup_batches))
	with torch.no_grad():
		it = iter(dl)
		for _ in range(warmup_batches):
			try:
				xyz, feature, indices, pts, _ = next(it)
			except StopIteration:
				it = iter(dl)
				xyz, feature, indices, pts, _ = next(it)
			xyz = xyz.to(device, non_blocking=True)
			feature = feature.to(device, non_blocking=True)
			indices = [ii.to(device, non_blocking=True).long() for ii in indices[::-1]]
			pts_list = pts.tolist()[::-1]
			with autocast(enabled=(device.type == "cuda")):
				_ = model(xyz, feature, indices, pts_list)

	# Measurement
	measure_batches = max(1, int(args.measure_batches))
	if args.mode == "forward":
		# Forward-only: exclude dataloading & H2D
		total_scenes = 0
		total_forward_time = 0.0
		with torch.no_grad():
			it = iter(dl)
			for _ in range(measure_batches):
				# Fetch next batch (I/O outside timing)
				try:
					xyz, feature, indices, pts, _ = next(it)
				except StopIteration:
					it = iter(dl)
					xyz, feature, indices, pts, _ = next(it)

				# Move tensors (H2D outside timing)
				xyz = xyz.to(device, non_blocking=True)
				feature = feature.to(device, non_blocking=True)
				indices = [ii.to(device, non_blocking=True).long() for ii in indices[::-1]]
				pts_list = pts.tolist()[::-1]

				# Time forward pass only
				if device.type == "cuda":
					torch.cuda.synchronize()
				t_start = time.perf_counter()
				with autocast(enabled=(device.type == "cuda")):
					_ = model(xyz, feature, indices, pts_list)
				if device.type == "cuda":
					torch.cuda.synchronize()
				t_total = time.perf_counter() - t_start
				total_forward_time += t_total

				# Count scenes
				B = int(pts.shape[1])
				total_scenes += B

		# Report forward-only throughput
		elapsed = max(1e-6, total_forward_time)
		scenes_per_sec = total_scenes / elapsed
		points_per_sec = scenes_per_sec * args.points
		avg_batch_latency_ms = 1000.0 * elapsed / measure_batches

		print("--- Throughput (forward-only) ---")
		print(f"Batch size (scenes): {args.batch_size}")
		print(f"Points per scene:    {args.points}")
		print(f"Measured batches:    {measure_batches}")
		print(f"Total scenes:        {total_scenes}")
		print(f"Forward time (s):    {elapsed:.4f}")
		print(f"Scenes / second:     {scenes_per_sec:.2f}")
		print(f"Points / second:     {points_per_sec:.0f}")
		print(f"Avg batch latency:   {avg_batch_latency_ms:.2f} ms")
	else:
		# Full pipeline: include dataloading & H2D moves
		total_scenes = 0
		if device.type == "cuda":
			# Ensure prior work done
			torch.cuda.synchronize()
		t_start = time.perf_counter()
		with torch.no_grad():
			it = iter(dl)
			for _ in range(measure_batches):
				try:
					xyz, feature, indices, pts, _ = next(it)
				except StopIteration:
					it = iter(dl)
					xyz, feature, indices, pts, _ = next(it)
				# Move + forward are part of timing in this mode
				xyz = xyz.to(device, non_blocking=True)
				feature = feature.to(device, non_blocking=True)
				indices = [ii.to(device, non_blocking=True).long() for ii in indices[::-1]]
				pts_list = pts.tolist()[::-1]
				with autocast(enabled=(device.type == "cuda")):
					_ = model(xyz, feature, indices, pts_list)
				B = int(pts.shape[1])
				total_scenes += B
		if device.type == "cuda":
			torch.cuda.synchronize()
		elapsed = max(1e-6, time.perf_counter() - t_start)
		scenes_per_sec = total_scenes / elapsed
		points_per_sec = scenes_per_sec * args.points
		avg_batch_latency_ms = 1000.0 * elapsed / measure_batches

		print("--- Throughput (full pipeline) ---")
		print(f"Batch size (scenes): {args.batch_size}")
		print(f"Points per scene:    {args.points}")
		print(f"Measured batches:    {measure_batches}")
		print(f"Total scenes:        {total_scenes}")
		print(f"Elapsed time (s):    {elapsed:.4f}")
		print(f"Scenes / second:     {scenes_per_sec:.2f}")
		print(f"Points / second:     {points_per_sec:.0f}")
		print(f"Avg batch latency:   {avg_batch_latency_ms:.2f} ms")


if __name__ == "__main__":
	main()

