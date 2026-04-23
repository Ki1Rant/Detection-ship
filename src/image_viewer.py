"""
Интерактивная утилита для просмотра и ручной очистки датасета.
GUI на Tkinter позволяет: просматривать изображения с разметкой, 
удалять некорректные образцы, навигировать по датасету и отслеживать прогресс.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import glob

DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset_finetune/first_data")

DATASET_SPLITS = ["train", "valid", "test"]

CLASS_NAMES = ["ship"]

WINDOW_WIDTH = 900
WINDOW_HEIGHT = 700



class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Box Viewer")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")

        self.dataset_path = DATASET_PATH
        self.splits = DATASET_SPLITS
        self.class_names = CLASS_NAMES

        self.current_split = None
        self.image_files = []
        self.current_index = 0
        self.current_image = None
        self.current_boxes = []

        self._build_ui()

    def _build_ui(self):
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill=tk.X)

        ttk.Label(top_frame, text="Dataset split:").pack(side=tk.LEFT)
        self.split_var = tk.StringVar(value=self.splits[0])
        self.split_cb = ttk.Combobox(
            top_frame, textvariable=self.split_var, values=self.splits,
            state="readonly", width=10
        )
        self.split_cb.pack(side=tk.LEFT, padx=5)

        self.load_btn = ttk.Button(top_frame, text="Load", command=self._load_split)
        self.load_btn.pack(side=tk.LEFT, padx=5)

        self.status_lbl = ttk.Label(top_frame, text="Выберите набор данных и нажмите Load", foreground="gray")
        self.status_lbl.pack(side=tk.LEFT, padx=10)

        self.canvas_frame = ttk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(self.canvas_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        btn_frame = ttk.Frame(self.root, padding=10)
        btn_frame.pack(fill=tk.X)

        self.prev_btn = ttk.Button(btn_frame, text="◀ Prev", command=self._prev)
        self.prev_btn.pack(side=tk.LEFT, padx=5)

        self.next_btn = ttk.Button(btn_frame, text="Next ▶", command=self._next)
        self.next_btn.pack(side=tk.LEFT, padx=5)

        self.keep_btn = ttk.Button(btn_frame, text="✓ Оставить", command=self._keep)
        self.keep_btn.pack(side=tk.LEFT, padx=20)

        self.delete_btn = ttk.Button(btn_frame, text="✗ Удалить", command=self._delete)
        self.delete_btn.pack(side=tk.LEFT, padx=5)

        self.lbl_info = ttk.Label(btn_frame, text="", foreground="blue")
        self.lbl_info.pack(side=tk.RIGHT, padx=10)

    def _load_split(self):
        split = self.split_var.get()
        self.current_split = split

        images_dir = os.path.join(self.dataset_path, split, "images")
        labels_dir = os.path.join(self.dataset_path, split, "labels")

        if not os.path.isdir(images_dir):
            messagebox.showerror("Error", f"Папка не найдена: {images_dir}")
            return

        self.image_files = sorted(
            glob.glob(os.path.join(images_dir, "*.jpg"))
            + glob.glob(os.path.join(images_dir, "*.png"))
            + glob.glob(os.path.join(images_dir, "*.jpeg"))
        )

        if not self.image_files:
            messagebox.showinfo("Info", "Изображения не найдены")
            self.status_lbl.config(text="Нет изображений")
            return

        self.current_index = 0
        self.status_lbl.config(text=f"Загружено: {len(self.image_files)} изображений ({split})")
        self._show_current()

    def _label_path(self, image_path):
        base = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(self.dataset_path, self.current_split, "labels", base + ".txt")

    def _parse_boxes(self, label_path):
        boxes = []
        if not os.path.exists(label_path):
            return boxes
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                x_center, y_center, w, h = map(float, parts[1:5])
                boxes.append((cls, x_center, y_center, w, h))
        return boxes

    def _show_current(self):
        if not self.image_files:
            return

        img_path = self.image_files[self.current_index]
        label_path = self._label_path(img_path)

        try:
            self.current_image = Image.open(img_path)
        except Exception as e:
            messagebox.showerror("Error", f"Не удалось открыть изображение:\n{img_path}\n{e}")
            return

        self.current_boxes = self._parse_boxes(label_path)

        # Draw boxes on image
        img_copy = self.current_image.copy()
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img_copy)

        img_w, img_h = img_copy.size
        for cls, xc, yc, w, h in self.current_boxes:
            x1 = int((xc - w / 2) * img_w)
            y1 = int((yc - h / 2) * img_h)
            x2 = int((xc + w / 2) * img_w)
            y2 = int((yc + h / 2) * img_h)
            class_name = self.class_names[cls] if cls < len(self.class_names) else str(cls)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1 - 10), class_name, fill="yellow")

        # Resize to fit canvas
        canvas_w = self.canvas.winfo_width() or 800
        canvas_h = self.canvas.winfo_height() or 550

        img_copy.thumbnail((canvas_w, canvas_h), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(img_copy)

        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_w // 2, canvas_h // 2,
            anchor=tk.CENTER, image=self.tk_image
        )

        boxes_count = len(self.current_boxes)
        self.lbl_info.config(
            text=f"[{self.current_index + 1}/{len(self.image_files)}] {os.path.basename(img_path)} | Boxes: {boxes_count}"
        )

    def _prev(self):
        if not self.image_files:
            return
        self.current_index = (self.current_index - 1) % len(self.image_files)
        self._show_current()

    def _next(self):
        if not self.image_files:
            return
        self.current_index = (self.current_index + 1) % len(self.image_files)
        self._show_current()

    def _keep(self):
        if not self.image_files:
            return
        self.status_lbl.config(
            text=f" Оставлено: {os.path.basename(self.image_files[self.current_index])}"
        )
        self._next()

    def _delete(self):
        if not self.image_files:
            return

        img_path = self.image_files[self.current_index]
        label_path = self._label_path(img_path)

        confirm = messagebox.askyesno(
            "Удалить",
            f"Удалить файл изображения и метки?\n\n{os.path.basename(img_path)}"
        )
        if not confirm:
            return

        try:
            if os.path.exists(img_path):
                os.remove(img_path)
            if os.path.exists(label_path):
                os.remove(label_path)
            self.image_files.pop(self.current_index)
            self.status_lbl.config(text=f"✗ Удалено: {os.path.basename(img_path)}")

            if not self.image_files:
                messagebox.showinfo("Info", "Все изображения удалены")
                self.lbl_info.config(text="Нет изображений")
                self.canvas.delete("all")
                return

            if self.current_index >= len(self.image_files):
                self.current_index = 0

            self._show_current()
        except Exception as e:
            messagebox.showerror("Error", f"Ошибка при удалении:\n{e}")


def main():
    root = tk.Tk()
    app = ImageViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
