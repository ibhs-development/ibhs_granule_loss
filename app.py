import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from loss import process_granule_loss


class GranuleLossApp:
    def __init__(self, root):
        self.root = root
        self.root.title("IBHS Granule Loss Analysis")
        self.root.geometry("1120x760")
        self.root.minsize(900, 620)

        # Variables
        self.input_folder_var = tk.StringVar()
        self.output_folder_var = tk.StringVar()
        self.threshold_var = tk.StringVar(value="2.58")
        self.status_var = tk.StringVar(value="Ready")

        # Track if processing is running
        self.is_processing = False

        # Create UI
        self.create_widgets()

    def create_widgets(self):
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Section.TLabel', font=('Arial', 10, 'bold'))
        style.configure('Accent.TButton', font=('Arial', 10, 'bold'))

        main_frame = ttk.Frame(self.root, padding="16")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(5, weight=1)

        ttk.Label(main_frame, text="IBHS Granule Loss Analysis", style='Title.TLabel').grid(
            row=0, column=0, sticky=tk.W
        )
        ttk.Label(
            main_frame,
            text="Select a folder of scale-bar images. Cropped analysis copies are saved beside each source image.",
        ).grid(row=1, column=0, sticky=tk.W, pady=(2, 12))

        folder_frame = ttk.LabelFrame(main_frame, text="Folders", padding="12")
        folder_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        folder_frame.columnconfigure(1, weight=1)

        ttk.Label(folder_frame, text="Scale-image folder", style='Section.TLabel').grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        ttk.Entry(folder_frame, textvariable=self.input_folder_var, width=60).grid(
            row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=8
        )
        ttk.Button(folder_frame, text="Browse...", command=self.browse_input_folder).grid(
            row=0, column=2, pady=5
        )

        ttk.Label(folder_frame, text="Results folder", style='Section.TLabel').grid(
            row=1, column=0, sticky=tk.W, pady=5
        )
        ttk.Entry(folder_frame, textvariable=self.output_folder_var, width=60).grid(
            row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=8
        )
        ttk.Button(folder_frame, text="Browse...", command=self.browse_output_folder).grid(
            row=1, column=2, pady=5
        )

        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="12")
        settings_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        settings_frame.columnconfigure(3, weight=1)

        ttk.Label(settings_frame, text="IGL/PGL threshold", style='Section.TLabel').grid(
            row=0, column=0, sticky=tk.W
        )
        ttk.Entry(settings_frame, textvariable=self.threshold_var, width=12).grid(
            row=0, column=1, sticky=tk.W, padx=(8, 4)
        )
        ttk.Label(settings_frame, text="mm2").grid(row=0, column=2, sticky=tk.W)
        ttk.Button(settings_frame, text="Reset to 2.58", command=lambda: self.threshold_var.set("2.58")).grid(
            row=0, column=3, sticky=tk.W, padx=(16, 0)
        )

        action_frame = ttk.Frame(main_frame)
        action_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        action_frame.columnconfigure(1, weight=1)

        self.run_button = ttk.Button(
            action_frame,
            text="Generate Crops + Analyze",
            command=self.run_analysis,
            style='Accent.TButton',
        )
        self.run_button.grid(row=0, column=0, sticky=tk.W)

        self.progress = ttk.Progressbar(action_frame, mode='indeterminate')
        self.progress.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=12)

        ttk.Label(action_frame, textvariable=self.status_var).grid(row=0, column=2, sticky=tk.E)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=5, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        log_frame = ttk.Frame(self.notebook)
        self.notebook.add(log_frame, text="Logs")
        self.log_text = scrolledtext.ScrolledText(
            log_frame, width=80, height=20, wrap=tk.WORD, font=('Courier', 9)
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        summary_frame = ttk.Frame(self.notebook, padding="5")
        self.notebook.add(summary_frame, text="Summary")
        summary_frame.columnconfigure(0, weight=1)
        summary_frame.rowconfigure(0, weight=1)

        columns = (
            "Impact",
            "Count_IGL",
            "Count_PGL",
            "AreaSum_IGL_mm2",
            "AreaSum_PGL_mm2",
            "GL_Rating",
            "CombinedGL_Rating",
        )
        self.summary_tree = ttk.Treeview(summary_frame, columns=columns, show="headings", height=12)
        for column in columns:
            self.summary_tree.heading(column, text=column)
            width = 150 if column == "Impact" else 120
            self.summary_tree.column(column, width=width, minwidth=80, anchor=tk.W)
        self.summary_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        summary_scroll = ttk.Scrollbar(summary_frame, orient=tk.VERTICAL, command=self.summary_tree.yview)
        summary_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.summary_tree.configure(yscrollcommand=summary_scroll.set)

        self.plot_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.plot_frame, text="Results Plot")

    def browse_input_folder(self):
        folder = filedialog.askdirectory(title="Select Input Folder")
        if folder:
            self.input_folder_var.set(folder)
            if not self.output_folder_var.get():
                self.output_folder_var.set(str(Path(folder) / "granule_loss_results"))

    def browse_output_folder(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder_var.set(folder)

    def log_message(self, message):
        """Add a message to the log text widget (thread-safe)."""
        self.root.after(0, self._append_log, message)

    def _append_log(self, message):
        """Internal method to append to log (must be called from main thread)."""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def validate_inputs(self):
        """Validate user inputs before running analysis."""
        if not self.input_folder_var.get():
            messagebox.showerror("Error", "Please select an input folder.")
            return False

        if not self.output_folder_var.get():
            self.output_folder_var.set(
                str(Path(self.input_folder_var.get()) / "granule_loss_results")
            )

        try:
            threshold = float(self.threshold_var.get())
            if threshold <= 0:
                raise ValueError("Threshold must be positive")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid positive number for the threshold.")
            return False

        if not Path(self.input_folder_var.get()).exists():
            messagebox.showerror("Error", "Input folder does not exist.")
            return False

        return True

    def run_analysis(self):
        """Run the granule loss analysis in a separate thread."""
        if self.is_processing:
            messagebox.showwarning("Warning", "Analysis is already running.")
            return

        if not self.validate_inputs():
            return

        # Clear previous logs and plot
        self.log_text.delete(1.0, tk.END)
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        for item in self.summary_tree.get_children():
            self.summary_tree.delete(item)

        # Start processing
        self.is_processing = True
        self.run_button.config(state='disabled')
        self.status_var.set("Running...")
        self.progress.start()

        # Switch to log tab
        self.notebook.select(0)

        # Run in separate thread
        thread = threading.Thread(target=self._run_analysis_thread, daemon=True)
        thread.start()

    def _run_analysis_thread(self):
        """Thread worker for running the analysis."""
        try:
            self.log_message("=" * 70)
            self.log_message("Starting Granule Loss Analysis...")
            self.log_message("=" * 70)

            input_folder = self.input_folder_var.get()
            output_folder = self.output_folder_var.get()
            threshold = float(self.threshold_var.get())

            self.log_message(f"\nInput Folder: {input_folder}")
            self.log_message(f"Output Folder: {output_folder}")
            self.log_message(f"IGL/PGL Threshold: {threshold} mm2\n")
            self.log_message("Cropped images will be saved beside each source as *_cropped.*\n")

            # Run the analysis
            summary_df, fig = process_granule_loss(
                input_folder=input_folder,
                output_folder=output_folder,
                igl_cutoff_mm2=threshold,
                log_callback=self.log_message
            )

            self.root.after(0, self._display_summary, summary_df)
            self.root.after(0, self._display_plot, fig)

            self.log_message("\n" + "=" * 70)
            self.log_message("Analysis completed successfully!")
            self.log_message("=" * 70)

            # Show success message
            self.root.after(0, messagebox.showinfo, "Success",
                          f"Analysis completed!\n\nResults saved to:\n{output_folder}")

        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}"
            self.log_message(f"\n{error_msg}")
            self.root.after(0, messagebox.showerror, "Error", error_msg)

        finally:
            # Re-enable UI
            self.root.after(0, self._finish_processing)

    def _display_summary(self, summary_df):
        """Display the output DataFrame in the summary tab."""
        for item in self.summary_tree.get_children():
            self.summary_tree.delete(item)

        columns = self.summary_tree["columns"]
        for _, row in summary_df.iterrows():
            values = []
            for column in columns:
                value = row[column]
                if isinstance(value, float):
                    value = f"{value:.3f}"
                values.append(value)
            self.summary_tree.insert("", tk.END, values=values)

    def _display_plot(self, fig):
        """Display the matplotlib figure in the plot tab."""
        # Clear previous plot
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # Create canvas with the figure
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add toolbar for plot interaction
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
        toolbar.update()

        # Switch to plot tab
        self.notebook.select(1)

    def _finish_processing(self):
        """Clean up after processing is complete."""
        self.progress.stop()
        self.run_button.config(state='normal')
        self.status_var.set("Ready")
        self.is_processing = False


def main():
    root = tk.Tk()
    app = GranuleLossApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
