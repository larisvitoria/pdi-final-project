import os
import sys
import subprocess
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText


class FilePicker(tk.Toplevel):
    def __init__(self, parent, title: str, initial_dir: str | None = None):
        super().__init__(parent)
        self.title(title)
        self.geometry("1200x700")
        self.result_path = None

        default_font = ("Helvetica", 16)
        mono_font = ("Consolas", 14)

        self.path_var = tk.StringVar(value=initial_dir or os.getcwd())

        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=10)
        ttk.Label(top, text="Folder:", width=10).pack(side="left")
        path_entry = ttk.Entry(top, textvariable=self.path_var, font=default_font)
        path_entry.pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(top, text="Up", command=self.go_up).pack(side="left", padx=4)
        ttk.Button(top, text="Open", command=self.refresh).pack(side="left", padx=4)

        search_frame = ttk.Frame(self)
        search_frame.pack(fill="x", padx=10, pady=4)
        ttk.Label(search_frame, text="Search:", width=10).pack(side="left")
        self.search_var = tk.StringVar(value="")
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, font=default_font)
        search_entry.pack(side="left", fill="x", expand=True)
        search_entry.bind("<KeyRelease>", lambda e: self.refresh())

        list_frame = ttk.Frame(self)
        list_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.listbox = tk.Listbox(list_frame, font=mono_font)
        self.listbox.pack(side="left", fill="both", expand=True)
        self.listbox.bind("<Double-1>", self.open_selected)

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=scrollbar.set)

        btns = ttk.Frame(self)
        btns.pack(fill="x", padx=10, pady=10)
        ttk.Button(btns, text="Select", command=self.select_current).pack(side="left")
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="left", padx=6)

        self.refresh()

    def go_up(self):
        cur = self.path_var.get()
        parent = os.path.dirname(cur.rstrip(os.sep))
        if parent and parent != cur:
            self.path_var.set(parent)
            self.refresh()

    def refresh(self):
        path = self.path_var.get()
        self.listbox.delete(0, tk.END)
        if not os.path.isdir(path):
            return
        query = self.search_var.get().lower().strip()
        entries = []
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if query and query not in name.lower():
                continue
            entries.append((name, full))
        for name, full in entries:
            tag = "[D] " if os.path.isdir(full) else "    "
            self.listbox.insert(tk.END, f"{tag}{name}")

    def open_selected(self, _=None):
        selection = self.listbox.curselection()
        if not selection:
            return
        name = self.listbox.get(selection[0]).strip()
        if name.startswith("[D]"):
            name = name.replace("[D]", "", 1).strip()
        path = os.path.join(self.path_var.get(), name)
        if os.path.isdir(path):
            self.path_var.set(path)
            self.refresh()
        else:
            self.result_path = path
            self.destroy()

    def select_current(self):
        selection = self.listbox.curselection()
        if not selection:
            return
        name = self.listbox.get(selection[0]).strip()
        if name.startswith("[D]"):
            name = name.replace("[D]", "", 1).strip()
        path = os.path.join(self.path_var.get(), name)
        if os.path.isdir(path):
            return
        self.result_path = path
        self.destroy()


def browse_file(entry: tk.Entry, title: str = "Select file"):
    picker = FilePicker(entry.winfo_toplevel(), title)
    entry.wait_window(picker)
    if picker.result_path:
        entry.delete(0, tk.END)
        entry.insert(0, picker.result_path)


def browse_save(entry: tk.Entry, title: str = "Select output file"):
    picker = FilePicker(entry.winfo_toplevel(), title)
    entry.wait_window(picker)
    if picker.result_path:
        entry.delete(0, tk.END)
        entry.insert(0, picker.result_path)


def run_inference(output_text: tk.Text, fields: dict):
    output_text.delete("1.0", tk.END)

    image = fields["image"].get().strip()
    view = fields["view"].get().strip()
    seg_models = fields["seg_models"].get().strip().split()
    seg_ckpts = fields["seg_ckpts"].get().strip().split()
    seg_threshold = fields["seg_threshold"].get().strip()
    clf_models = fields["clf_models"].get().strip().split()
    clf_ckpts = fields["clf_ckpts"].get().strip().split()
    clf_threshold = fields["clf_threshold"].get().strip()
    overlay_path = fields["overlay_path"].get().strip()

    if not image:
        output_text.insert(tk.END, "Please select an image.\n")
        return

    if len(seg_models) != len(seg_ckpts):
        output_text.insert(tk.END, "Seg models and ckpts counts do not match.\n")
        return

    if len(clf_models) != len(clf_ckpts):
        output_text.insert(tk.END, "Classifier models and ckpts counts do not match.\n")
        return

    cmd = [sys.executable, "single_image_infer.py", "--image", image]

    if view:
        cmd += ["--view", view]

    cmd += ["--seg_models", *seg_models]
    cmd += ["--seg_ckpts", *seg_ckpts]
    cmd += ["--seg_threshold", seg_threshold]
    cmd += ["--clf_models", *clf_models]
    cmd += ["--clf_ckpts", *clf_ckpts]
    cmd += ["--clf_threshold", clf_threshold]

    if overlay_path:
        cmd += ["--overlay_path", overlay_path]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        output_text.insert(tk.END, result.stdout)
        if result.stderr:
            output_text.insert(tk.END, "\n[stderr]\n" + result.stderr)
    except Exception as exc:
        output_text.insert(tk.END, f"Error running inference: {exc}\n")


def main():
    root = tk.Tk()
    root.title("CBIS-DDSM Single Image Inference")
    root.geometry("2600x1800")

    default_font = ("Helvetica", 24)
    header_font = ("Helvetica", 30, "bold")
    mono_font = ("Consolas", 22)

    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    style.configure(".", font=default_font)
    style.configure("Header.TLabel", font=header_font)
    style.configure("TButton", font=default_font)
    style.configure("TEntry", font=default_font)

    fields = {}

    def add_field(label: str, default: str = "", browse: bool = False, save: bool = False):
        frame = ttk.Frame(root)
        frame.pack(fill="x", padx=10, pady=4)
        ttk.Label(frame, text=label, width=24).pack(side="left")
        entry = ttk.Entry(frame, font=default_font)
        entry.pack(side="left", fill="x", expand=True)
        if default:
            entry.insert(0, default)
        if browse:
            ttk.Button(frame, text="Browse", command=lambda: browse_file(entry)).pack(side="left", padx=5)
        if save:
            ttk.Button(frame, text="Pick File", command=lambda: browse_save(entry)).pack(side="left", padx=5)
        fields[label] = entry

    ttk.Label(root, text="Inputs", style="Header.TLabel").pack(anchor="w", padx=10, pady=(10, 0))
    add_field("image", browse=True)
    add_field("view", default="CC")
    ttk.Label(root, text="Segmentation", style="Header.TLabel").pack(anchor="w", padx=10, pady=(10, 0))
    add_field("seg_models", default="deeplabv3p")
    add_field("seg_ckpts", default="seg_runs_mass_new/deeplabv3p/deeplabv3p_best.pt")
    add_field("seg_threshold", default="0.5")
    ttk.Label(root, text="Classification", style="Header.TLabel").pack(anchor="w", padx=10, pady=(10, 0))
    add_field("clf_models", default="resnext50_32x4d densenet121 efficientnet_v2_s")
    add_field("clf_ckpts", default="resnext50_32x4d_best.pth densenet121_best.pth efficientnet_v2_s_best.pth")
    add_field("clf_threshold", default="0.70")
    ttk.Label(root, text="Outputs", style="Header.TLabel").pack(anchor="w", padx=10, pady=(10, 0))
    add_field("overlay_path", default="single_infer_overlay.png", save=True)

    btn_frame = ttk.Frame(root)
    btn_frame.pack(fill="x", padx=10, pady=8)
    ttk.Button(btn_frame, text="Run Inference", command=lambda: run_inference(output_text, fields)).pack(side="left")

    ttk.Label(root, text="Output", style="Header.TLabel").pack(anchor="w", padx=10, pady=(10, 0))
    output_text = ScrolledText(root, height=20, font=mono_font, wrap="word")
    output_text.pack(fill="both", expand=True, padx=10, pady=10)

    # Map field keys to entries for run_inference
    mapped = {
        "image": fields["image"],
        "view": fields["view"],
        "seg_models": fields["seg_models"],
        "seg_ckpts": fields["seg_ckpts"],
        "seg_threshold": fields["seg_threshold"],
        "clf_models": fields["clf_models"],
        "clf_ckpts": fields["clf_ckpts"],
        "clf_threshold": fields["clf_threshold"],
        "overlay_path": fields["overlay_path"],
    }

    # Override run_inference to use mapped
    def run_button():
        run_inference(output_text, mapped)

    for child in btn_frame.winfo_children():
        child.destroy()
    ttk.Button(btn_frame, text="Run Inference", command=run_button).pack(side="left")

    root.mainloop()


if __name__ == "__main__":
    main()
