import asyncio
import os
import cv2
import numpy as np
from pyppeteer import launch

# List of HTML files to convert (you can add/remove as needed)
html_files = [
    f"C:/Users/akkis/OneDrive/Desktop/Boxing_Analysis_Project/outputs/boxing_analysis_3d_video_{i}.html" for i in range(1, 6)
]

# Folder to save the video outputs
output_folder = "video_outputs"
os.makedirs(output_folder, exist_ok=True)

async def convert_html_to_video(html_path, video_path, duration=10, fps=15):
    print(f"🎬 Converting: {html_path} → {video_path}")
    browser = await launch(
    headless=True,
    executablePath="C:/Program Files/Google/Chrome/Application/chrome.exe",  # <-- adjust if needed
    args=["--no-sandbox"]
    )

    page = await browser.newPage()
    await page.setViewport({'width': 1280, 'height': 720})
    await page.goto(f"file:///{os.path.abspath(html_path)}")
    await page.waitForSelector('.main-svg')

    frames = []
    for _ in range(duration * fps):
        screenshot = await page.screenshot()
        img_array = np.frombuffer(screenshot, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        frames.append(frame)
        await asyncio.sleep(1 / fps)

    await browser.close()

    # Save video
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"✅ Saved video to: {video_path}\n")

async def main():
    tasks = []
    for html_file in html_files:
        if os.path.exists(html_file):
            output_video = os.path.join(output_folder, html_file.replace(".html", ".mp4"))
            tasks.append(convert_html_to_video(html_file, output_video))
        else:
            print(f"⚠️ Skipping {html_file} - File not found.")
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
