import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import os


class CSVNARemoverApp:
    def __init__(self, master, subset_files_function, default_folder):
        self.master = master
        self.master.title("CSV NA Remover")
        self.master.geometry("800x600")
        self.master.resizable(True, True)

        # Style configuration
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", padding=6, font=("Arial", 10), background="#f0f0f0")
        style.configure("TLabel", padding=6, font=("Arial", 11), background="#eaeaea")
        style.configure("TFrame", background="#f7f7f7")
        style.configure("Treeview", background="#ffffff", foreground="#000000", fieldbackground="#ffffff")
        style.configure("Treeview.Heading", font=("Arial", 10, "bold"), background="#dcdcdc")

        self.file_path = None
        self.df = None
        self.columns = []
        self.get_subset_files = subset_files_function
        self.default_folder = default_folder

        # Main layout using Grid
        self.main_frame = ttk.Frame(self.master)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)

        # File selection
        ttk.Label(self.main_frame, text="Choose a CSV File:", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky="w", pady=5)

        self.file_list_frame = ttk.Frame(self.main_frame)
        self.file_list_frame.grid(row=1, column=0, sticky="nsew", pady=5)

        self.file_listbox = tk.Listbox(self.file_list_frame, height=8, selectmode=tk.SINGLE, activestyle="dotbox")
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.file_scrollbar = ttk.Scrollbar(self.file_list_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        self.file_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.file_listbox.config(yscrollcommand=self.file_scrollbar.set)

        self.load_files_from_default_folder()

        ttk.Button(self.main_frame, text="Select File", command=self.select_file).grid(row=2, column=0, pady=5, sticky="ew")

        # Columns selection
        ttk.Label(self.main_frame, text="Select Columns to Check for NAs:", font=("Arial", 12, "bold")).grid(row=3, column=0, sticky="w", pady=5)

        self.columns_frame = ttk.Frame(self.main_frame)
        self.columns_frame.grid(row=4, column=0, sticky="nsew", pady=5)

        self.treeview = ttk.Treeview(self.columns_frame, columns=("Columns"), show="headings", selectmode="extended")
        self.treeview.heading("Columns", text="Column Name")
        self.treeview.column("Columns", anchor="w", width=300)

        self.treeview_scrollbar = ttk.Scrollbar(self.columns_frame, orient=tk.VERTICAL, command=self.treeview.yview)
        self.treeview.config(yscrollcommand=self.treeview_scrollbar.set)

        self.treeview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.treeview_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.all_columns_var = tk.BooleanVar()
        self.all_columns_checkbox = ttk.Checkbutton(self.main_frame, text="Select All Columns", variable=self.all_columns_var, command=self.toggle_all_columns)
        self.all_columns_checkbox.grid(row=5, column=0, pady=5, sticky="w")

        self.remove_button = ttk.Button(self.main_frame, text="Remove NAs", command=self.remove_na, state=tk.DISABLED)
        self.remove_button.grid(row=6, column=0, pady=10, sticky="ew")

        # Dynamic resizing
        self.main_frame.rowconfigure(1, weight=1)
        self.main_frame.rowconfigure(4, weight=3)  # Ensures the columns section grows
        self.main_frame.columnconfigure(0, weight=1)

    def load_files_from_default_folder(self):
        try:
            subset_files = self.get_subset_files(self.default_folder, ".csv")
            if not subset_files:
                messagebox.showwarning("Warning", "No CSV files found in the default folder!")
                return

            self.file_paths = {os.path.basename(f): f for f in subset_files}
            self.file_listbox.delete(0, tk.END)
            for file_name in self.file_paths.keys():
                self.file_listbox.insert(tk.END, file_name)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load files from default folder: {e}")

    def select_file(self):
        selection = self.file_listbox.curselection()
        if selection:
            selected_file = self.file_listbox.get(selection[0])
            self.file_path = self.file_paths[selected_file]
            try:
                self.df = pd.read_csv(self.file_path)
                self.columns = self.df.columns.tolist()
                self.populate_columns()
                self.remove_button.config(state=tk.NORMAL)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")
        else:
            messagebox.showwarning("Warning", "No file selected!")

    def populate_columns(self):
        self.treeview.delete(*self.treeview.get_children())
        for col in self.columns:
            self.treeview.insert("", "end", values=(col,))

    def toggle_all_columns(self):
        if self.all_columns_var.get():
            self.treeview.selection_set(self.treeview.get_children())
        else:
            self.treeview.selection_remove(self.treeview.get_children())

    def remove_na(self):
        try:
            selected_items = self.treeview.selection()
            selected_columns = [self.treeview.item(item, "values")[0] for item in selected_items]
            if self.all_columns_var.get() or not selected_columns:
                selected_columns = self.columns

            if not selected_columns:
                messagebox.showwarning("Warning", "No columns selected!")
                return

            initial_row_count = len(self.df)
            filtered_df = self.df.dropna(subset=selected_columns)
            removed_row_count = initial_row_count - len(filtered_df)

            column_na_counts = self.df[selected_columns].isna().sum()
            details = "\n".join([f"{col}: {count}" for col, count in column_na_counts.items() if count > 0])

            save_path = self.file_path.replace(".csv", "_noNA.csv")
            filtered_df.to_csv(save_path, index=False)

            messagebox.showinfo(
                "Success",
                f"File saved as {save_path}!\nTotal rows removed: {removed_row_count}\n\nDetails:\n{details}"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to remove NAs: {e}")


if __name__ == "__main__":
    def csv_get_files_in_subfolders(folder, extension):
        subset_files = []
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(extension):
                    subset_files.append(os.path.join(root, file))
        return subset_files

    DEFAULT_FOLDER = "subsets_by_date"  # Default folder path

    root = tk.Tk()
    app = CSVNARemoverApp(root, csv_get_files_in_subfolders, DEFAULT_FOLDER)
    root.mainloop()
