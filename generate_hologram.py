import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def generate_serialized_holograms(output_dir="batch_frames", count=10):
    os.makedirs(output_dir, exist_ok=True)

    for i in range(count):
        # 고정 배경 생성
        base = np.zeros((400, 400, 3), dtype=np.uint8)
        for y in range(400):
            for x in range(400):
                base[y, x] = ((x + 100) % 255, (y + 50) % 255, (x + y) % 255)

        # 로고 삽입
        logo = cv2.imread("assets/logo.png", cv2.IMREAD_UNCHANGED)
        if logo is not None and logo.shape[2] == 4:
            logo = cv2.resize(logo, (100, 100))
            x_offset, y_offset = 150, 150
            for c in range(3):
                base[y_offset:y_offset+logo.shape[0], x_offset:x_offset+logo.shape[1], c] = \
                    logo[:, :, c] * (logo[:, :, 3] / 255.0) + \
                    base[y_offset:y_offset+logo.shape[0], x_offset:x_offset+logo.shape[1], c] * (1.0 - logo[:, :, 3] / 255.0)

        # 텍스트 삽입 (PIL)
        pil_img = Image.fromarray(base)
        draw = ImageDraw.Draw(pil_img)

        try:
            font = ImageFont.truetype("assets/microtext.ttf", 8)
            font_big = ImageFont.truetype("assets/microtext.ttf", 24)
        except:
            font = ImageFont.load_default()
            font_big = ImageFont.load_default()

        # 마이크로 텍스트 반복
        for x in range(0, 400, 40):
            draw.text((x, 380), "AUTH-ECOQCODE", font=font, fill=(255, 255, 255, 128))

        # 고유 시리얼 넘버 삽입
        serial = f"ID: ECOQ-{1000 + i}"
        draw.text((100, 50), serial, font=font_big, fill=(200, 200, 255, 180))

        # 저장
        final_img = np.array(pil_img)
        out_path = os.path.join(output_dir, f"hologram_{i:02d}.png")
        cv2.imwrite(out_path, final_img)
        print(f"✅ Saved {out_path} with serial '{serial}'")

def combine_serial_images_to_video(input_dir="batch_frames", output="serialized_holograms.mp4", fps=2):
    import glob

    frame_files = sorted(glob.glob(os.path.join(input_dir, "hologram_*.png")))
    if not frame_files:
        print("⚠️ No frames found.")
        return

    first_frame = cv2.imread(frame_files[0])
    h, w, _ = first_frame.shape

    out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frame_files:
        frame = cv2.imread(f)
        out.write(frame)
    out.release()
    print(f"🎥 Video saved to {output}")


if __name__ == "__main__":
    generate_serialized_holograms(count=20)
    combine_serial_images_to_video()
