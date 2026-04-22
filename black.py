import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os


# -------------------------------
# Detect BLACK boxes
# -------------------------------
def find_black_boxes(template, min_area=1000, debug=False):
    gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Detect DARK regions instead of white
    _, th = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        crop = gray[y:y+h, x:x+w]

        # Ensure it's actually dark
        if cv2.mean(crop)[0] > 80:
            continue

        boxes.append((x, y, w, h))

    # Sort top → bottom
    boxes.sort(key=lambda r: (r[1], r[0]))

    if debug:
        print(f"Detected {len(boxes)} black boxes:")
        for b in boxes:
            print(b)

    return boxes


# -------------------------------
# Apply filters
# -------------------------------
def apply_filters(image):
    image = ImageEnhance.Contrast(image).enhance(1.1)
    image = ImageEnhance.Color(image).enhance(1.05)
    return image


# -------------------------------
# Resize & place image
# -------------------------------
def place_image(template, photo_pil, box, mode='cover', bw=False):
    x, y, w, h = box

    photo_cv = cv2.cvtColor(np.array(photo_pil), cv2.COLOR_RGB2BGR)
    src_h, src_w = photo_cv.shape[:2]

    scale = max(w / src_w, h / src_h) if mode == 'cover' else min(w / src_w, h / src_h)

    new_w = int(src_w * scale)
    new_h = int(src_h * scale)

    resized = cv2.resize(photo_cv, (new_w, new_h),
                         interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)

    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    if mode == 'cover':
        start_x = (new_w - w) // 2
        start_y = (new_h - h) // 2
        cropped = resized[start_y:start_y+h, start_x:start_x+w]
        canvas = cropped
    else:
        offset_x = (w - new_w) // 2
        offset_y = (h - new_h) // 2
        canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized

    placed_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

    if bw:
        placed_pil = placed_pil.convert('L').convert('RGB')
    else:
        placed_pil = apply_filters(placed_pil)

    final = cv2.cvtColor(np.array(placed_pil), cv2.COLOR_RGB2BGR)
    template[y:y+h, x:x+w] = final

    return template


# -------------------------------
# Main function
# -------------------------------
def fill_template(photo_paths, template_path, output_path="output.jpg", debug=False):

    if not os.path.exists(template_path):
        print("Template not found")
        return

    template = cv2.imread(template_path)

    if template is None:
        print("Failed to load template")
        return

    boxes = find_black_boxes(template, debug=debug)

    if not boxes:
        print("No black boxes detected")
        return

    result = template.copy()

    count = min(len(photo_paths), len(boxes))

    for i in range(count):
        path = photo_paths[i]

        if not os.path.exists(path):
            print(f"Missing image: {path}")
            continue

        img = Image.open(path).convert("RGB")

        if debug:
            print(f"Placing {path} -> {boxes[i]}")

        result = place_image(result, img, boxes[i])

    final = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    final.save(output_path, quality=100)

    print(f"✅ Saved: {output_path}")


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":

    template_path = "T.png"

    photo_paths = [
        "E.JPG"
    ]

    fill_template(
        photo_paths=photo_paths,
        template_path=template_path,
        output_path="final_output.jpg",
        debug=True
    )