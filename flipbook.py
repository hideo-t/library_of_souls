#!/usr/bin/env python3
"""
パラパラ漫画風動画生成スクリプト
静止画を合成して、パラパラ漫画風に再生し、最後の1枚で止める

Usage:
  python flipbook.py --input assets/illustrations/stories --output flipbook.mp4
  python flipbook.py --input assets/illustrations/stories --output flipbook.gif --format gif
  python flipbook.py --input assets/illustrations/stories --output flipbook.mp4 --fps 4 --hold 3
"""

import os
import sys
import argparse
import glob
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
except ImportError:
    print("PIL未インストール: pip install Pillow --break-system-packages")
    sys.exit(1)

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("⚠ OpenCV未インストール（GIF出力は可能、MP4には必要）")
    print("  pip install opencv-python numpy --break-system-packages")


def load_images(input_path, canvas_size=(1080, 1920)):
    """画像を読み込み、キャンバスサイズにリサイズ"""
    supported = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    files = []
    for ext in supported:
        files.extend(glob.glob(os.path.join(input_path, ext)))
    files.sort()

    if not files:
        print(f"❌ 画像が見つかりません: {input_path}")
        sys.exit(1)

    print(f"📁 {len(files)} 枚の画像を検出")
    images = []
    for f in files:
        img = Image.open(f).convert("RGB")
        img = fit_to_canvas(img, canvas_size)
        images.append((img, Path(f).stem))
        print(f"  ✅ {Path(f).name}")

    return images


def fit_to_canvas(img, canvas_size):
    """アスペクト比を保ちつつキャンバスにフィット（黒帯なし・クロップ）"""
    cw, ch = canvas_size
    iw, ih = img.size

    # カバーフィット（余白なし）
    scale = max(cw / iw, ch / ih)
    new_w = int(iw * scale)
    new_h = int(ih * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    # 中央クロップ
    left = (new_w - cw) // 2
    top = (new_h - ch) // 2
    img = img.crop((left, top, left + cw, top + ch))

    return img


# ============================================================
# エフェクト関数群
# ============================================================

def effect_none(img):
    """エフェクトなし"""
    return img


def effect_page_turn(current, next_img, progress, canvas_size):
    """ページめくり風トランジション"""
    cw, ch = canvas_size
    result = current.copy()

    # 右からスライドイン
    offset = int(cw * (1 - progress))
    result.paste(next_img, (offset, 0))

    # めくり線（影）
    if 0.05 < progress < 0.95:
        draw = ImageDraw.Draw(result)
        line_x = offset
        for i in range(20):
            alpha = int(80 * (1 - i / 20))
            draw.line(
                [(line_x - i, 0), (line_x - i, ch)],
                fill=(0, 0, 0), width=1
            )

    return result


def effect_dissolve(current, next_img, progress):
    """ディゾルブ（クロスフェード）"""
    return Image.blend(current, next_img, progress)


def effect_flash(img, intensity=0.8):
    """フラッシュ白飛び"""
    white = Image.new("RGB", img.size, (255, 255, 255))
    return Image.blend(img, white, intensity)


def effect_zoom_in(img, scale=1.05):
    """わずかにズームイン"""
    w, h = img.size
    new_w = int(w * scale)
    new_h = int(h * scale)
    zoomed = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - w) // 2
    top = (new_h - h) // 2
    return zoomed.crop((left, top, left + w, top + h))


def effect_slight_shake(img, offset=3):
    """微振動"""
    import random
    dx = random.randint(-offset, offset)
    dy = random.randint(-offset, offset)
    w, h = img.size
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    canvas.paste(img, (dx, dy))
    return canvas


# ============================================================
# フレーム生成
# ============================================================

def generate_frames(images, fps=6, hold_sec=3, transition_frames=4,
                    canvas_size=(1080, 1920), style="flipbook"):
    """
    パラパラ漫画風フレーム列を生成

    Args:
        images: [(PIL.Image, name), ...]
        fps: フレームレート
        hold_sec: 最後の画像で止まる秒数
        transition_frames: 各画像間のトランジションフレーム数
        canvas_size: 出力サイズ
        style: "flipbook" | "smooth" | "dramatic"
    """
    frames = []
    total = len(images)

    for i, (img, name) in enumerate(images):
        is_last = (i == total - 1)
        print(f"  🎬 [{i+1}/{total}] {name}", end="")

        if style == "flipbook":
            # === パラパラ漫画風 ===
            # 各画像を数フレーム表示（パラパラ感）
            display_frames = 2 if not is_last else int(fps * hold_sec)

            for f in range(display_frames):
                frame = img.copy()
                # 最初のフレームで軽い振動（パラパラ感演出）
                if f == 0 and not is_last:
                    frame = effect_slight_shake(frame, offset=2)
                frames.append(frame)

            # 次の画像へのフラッシュ（最後以外）
            if not is_last:
                flash = effect_flash(img, intensity=0.3)
                frames.append(flash)

        elif style == "smooth":
            # === スムーズトランジション ===
            # 各画像を一定時間表示
            display_frames = max(3, fps // 2) if not is_last else int(fps * hold_sec)
            for _ in range(display_frames):
                frames.append(img.copy())

            # ディゾルブトランジション
            if not is_last:
                next_img = images[i + 1][0]
                for t in range(transition_frames):
                    progress = (t + 1) / (transition_frames + 1)
                    blended = effect_dissolve(img, next_img, progress)
                    frames.append(blended)

        elif style == "dramatic":
            # === ドラマチック（ズーム+ページめくり） ===
            display_frames = max(4, fps // 2) if not is_last else int(fps * hold_sec)

            for f in range(display_frames):
                frame = img.copy()
                # 表示中にゆっくりズーム
                zoom = 1.0 + (f / display_frames) * 0.03
                frame = effect_zoom_in(frame, scale=zoom)
                frames.append(frame)

            # ページめくりトランジション
            if not is_last:
                next_img = images[i + 1][0]
                for t in range(transition_frames * 2):
                    progress = (t + 1) / (transition_frames * 2 + 1)
                    turned = effect_page_turn(img, next_img, progress, canvas_size)
                    frames.append(turned)

        print(f" → {len(frames)} frames")

    return frames


# ============================================================
# 出力
# ============================================================

def save_as_gif(frames, output_path, fps=6, hold_sec=3):
    """GIF形式で保存"""
    print(f"\n💾 GIF保存中: {output_path}")

    # フレーム間隔（ミリ秒）
    duration_normal = int(1000 / fps)
    # 最後のフレームはhold_sec分
    durations = [duration_normal] * len(frames)
    durations[-1] = hold_sec * 1000

    # サイズを縮小（GIFは重いので）
    max_gif_width = 540
    if frames[0].size[0] > max_gif_width:
        ratio = max_gif_width / frames[0].size[0]
        new_size = (max_gif_width, int(frames[0].size[1] * ratio))
        frames = [f.resize(new_size, Image.LANCZOS) for f in frames]
        print(f"  GIF用にリサイズ: {new_size[0]}x{new_size[1]}")

    # 減色（GIF最適化）
    frames_p = [f.quantize(colors=128, method=Image.MEDIANCUT).convert("RGB")
                for f in frames]

    frames_p[0].save(
        output_path,
        save_all=True,
        append_images=frames_p[1:],
        duration=durations,
        loop=0,
        optimize=True
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  ✅ 完了: {size_mb:.1f} MB")


def save_as_mp4(frames, output_path, fps=6):
    """MP4形式で保存（OpenCV必要）"""
    if not HAS_CV2:
        print("❌ OpenCVが必要です: pip install opencv-python")
        return

    print(f"\n💾 MP4保存中: {output_path}")

    h, w = frames[0].size[1], frames[0].size[0]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for i, frame in enumerate(frames):
        # PIL → OpenCV (RGB→BGR)
        cv_frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        out.write(cv_frame)

    out.release()

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  ✅ 完了: {size_mb:.1f} MB ({len(frames)} frames)")

    # ffmpeg があれば再エンコード（互換性向上）
    try:
        import subprocess
        temp = output_path + ".temp.mp4"
        os.rename(output_path, temp)
        subprocess.run([
            "ffmpeg", "-y", "-i", temp,
            "-c:v", "libx264", "-preset", "medium",
            "-crf", "23", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path
        ], capture_output=True)

        if os.path.exists(output_path):
            os.remove(temp)
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  🎥 ffmpeg再エンコード完了: {size_mb:.1f} MB")
        else:
            os.rename(temp, output_path)
    except FileNotFoundError:
        print("  ⚠ ffmpeg未インストール（mp4vコーデックで出力）")


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="静止画からパラパラ漫画風動画を生成"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="入力画像フォルダ"
    )
    parser.add_argument(
        "--output", "-o", default="flipbook.mp4",
        help="出力ファイルパス (default: flipbook.mp4)"
    )
    parser.add_argument(
        "--fps", type=int, default=6,
        help="フレームレート (default: 6)"
    )
    parser.add_argument(
        "--hold", type=float, default=3.0,
        help="最後の画像で止まる秒数 (default: 3.0)"
    )
    parser.add_argument(
        "--style", choices=["flipbook", "smooth", "dramatic"],
        default="flipbook",
        help="アニメーションスタイル (default: flipbook)"
    )
    parser.add_argument(
        "--size", default="1080x1920",
        help="出力サイズ WxH (default: 1080x1920 / スマホ縦)"
    )
    parser.add_argument(
        "--format", choices=["mp4", "gif", "both"],
        default="mp4",
        help="出力形式 (default: mp4)"
    )
    parser.add_argument(
        "--transition", type=int, default=4,
        help="トランジションフレーム数 (default: 4)"
    )

    args = parser.parse_args()

    # キャンバスサイズ
    w, h = map(int, args.size.split("x"))
    canvas_size = (w, h)

    print("=" * 50)
    print("📖 パラパラ漫画ジェネレーター")
    print("=" * 50)
    print(f"  入力:   {args.input}")
    print(f"  出力:   {args.output}")
    print(f"  FPS:    {args.fps}")
    print(f"  最後:   {args.hold}秒停止")
    print(f"  スタイル: {args.style}")
    print(f"  サイズ: {w}x{h}")
    print()

    # 画像読み込み
    images = load_images(args.input, canvas_size)

    # フレーム生成
    print(f"\n🎬 フレーム生成中 (style: {args.style})...")
    frames = generate_frames(
        images,
        fps=args.fps,
        hold_sec=args.hold,
        transition_frames=args.transition,
        canvas_size=canvas_size,
        style=args.style
    )
    print(f"\n  合計: {len(frames)} フレーム")

    # 出力
    output_path = Path(args.output)

    if args.format in ("mp4", "both"):
        mp4_path = output_path.with_suffix(".mp4")
        save_as_mp4(frames, str(mp4_path), fps=args.fps)

    if args.format in ("gif", "both"):
        gif_path = output_path.with_suffix(".gif")
        save_as_gif(frames, str(gif_path), fps=args.fps, hold_sec=args.hold)

    print("\n🎉 完了!")


if __name__ == "__main__":
    main()
