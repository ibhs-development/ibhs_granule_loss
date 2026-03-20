import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from loss import process_granule_loss


class GranuleLossApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Granule Loss Analysis")
        self.root.geometry("1000x700")

        # Variables
        self.input_folder_var = tk.StringVar()
        self.output_folder_var = tk.StringVar()
        self.threshold_var = tk.StringVar(value="2.58")

        # Track if processing is running
        self.is_processing = False

        # Create UI
        self.create_widgets()

    def create_widgets(self):
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)

        # Input Folder Selection
        ttk.Label(main_frame, text="Input Folder:", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        ttk.Entry(main_frame, textvariable=self.input_folder_var, width=50).grid(
            row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=5
        )
        ttk.Button(main_frame, text="Browse...", command=self.browse_input_folder).grid(
            row=0, column=2, pady=5
        )

        # Output Folder Selection
        ttk.Label(main_frame, text="Output Folder:", font=('Arial', 10, 'bold')).grid(
            row=1, column=0, sticky=tk.W, pady=5
        )
        ttk.Entry(main_frame, textvariable=self.output_folder_var, width=50).grid(
            row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=5
        )
        ttk.Button(main_frame, text="Browse...", command=self.browse_output_folder).grid(
            row=1, column=2, pady=5
        )

        # IGL/PGL Threshold
        threshold_frame = ttk.Frame(main_frame)
        threshold_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)

        ttk.Label(threshold_frame, text="IGL vs PGL Threshold (mm2):", font=('Arial', 10, 'bold')).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        ttk.Entry(threshold_frame, textvariable=self.threshold_var, width=15).pack(side=tk.LEFT)
        ttk.Label(threshold_frame, text="(Default: 2.58 mm2)", font=('Arial', 9, 'italic')).pack(
            side=tk.LEFT, padx=(10, 0)
        )

        # Run Button
        self.run_button = ttk.Button(
            main_frame, text="Run Analysis", command=self.run_analysis, style='Accent.TButton'
        )
        self.run_button.grid(row=3, column=0, columnspan=3, pady=15)

        # Progress Bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

        # Notebook for Logs and Plot
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        # Log Tab
        log_frame = ttk.Frame(self.notebook)
        self.notebook.add(log_frame, text="Logs")

        self.log_text = scrolledtext.ScrolledText(
            log_frame, width=80, height=20, wrap=tk.WORD, font=('Courier', 9)
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Plot Tab
        self.plot_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.plot_frame, text="Results Plot")

        # Configure style
        style = ttk.Style()
        style.configure('Accent.TButton', font=('Arial', 10, 'bold'))

    def browse_input_folder(self):
        folder = filedialog.askdirectory(title="Select Input Folder")
        if folder:
            self.input_folder_var.set(folder)

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
            messagebox.showerror("Error", "Please select an output folder.")
            return False

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

        # Start processing
        self.is_processing = True
        self.run_button.config(state='disabled')
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

            # Run the analysis
            summary_df, fig = process_granule_loss(
                input_folder=input_folder,
                output_folder=output_folder,
                igl_cutoff_mm2=threshold,
                log_callback=self.log_message
            )

            # Display the plot
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
        self.is_processing = False


def main():
    root = tk.Tk()
    app = GranuleLossApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
