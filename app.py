import argparse
import contextlib
import csv
import queue
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from deformation_analyzer import (
    build_parser as build_deformation_parser,
    resolve_tesseract_command,
    run as run_deformation_analysis,
    validate_args as validate_deformation_args,
)
from loss import SCALE_BAR_CANDIDATES_MM, process_granule_loss

SCALE_AUTO = "Auto-detect"
# Derived from the pipeline's own candidate list so the GUI cannot offer a length
# the reader does not know about, or miss one it does.
SCALE_CHOICES = (SCALE_AUTO, *(f"{mm} mm" for mm in SCALE_BAR_CANDIDATES_MM))
SCALE_TAGS = "/".join(f"'{mm}mm'" for mm in SCALE_BAR_CANDIDATES_MM)

DEFORMATION_SCALE_OCR = "OCR from scale images"
DEFORMATION_SCALE_SQUARE = "Fixed square mm/pixel"
DEFORMATION_SCALE_XY = "Fixed X/Y mm/pixel"
DEFORMATION_SCALE_MODES = (
    DEFORMATION_SCALE_OCR,
    DEFORMATION_SCALE_SQUARE,
    DEFORMATION_SCALE_XY,
)

# Read defaults from deformation_analyzer's own parser so its CLI and GUI stay
# aligned if a default is adjusted later.
_DEFORMATION_PARSER = build_deformation_parser()


def _deformation_default(name):
    return _DEFORMATION_PARSER.get_default(name)


class _LineLogWriter:
    """Turn print output from the analyzer into complete GUI log lines."""

    def __init__(self, callback):
        self.callback = callback
        self.buffer = ""

    def write(self, value):
        self.buffer += str(value)
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self.callback(line)
        return len(value)

    def flush(self):
        if self.buffer:
            self.callback(self.buffer)
            self.buffer = ""


class GranuleLossApp:
    def __init__(self, root):
        self.root = root
        self.root.title("IBHS Image Analysis")
        self.root.geometry("1180x820")
        self.root.minsize(900, 620)

        # Variables
        self.input_folder_var = tk.StringVar()
        self.output_folder_var = tk.StringVar()
        self.threshold_var = tk.StringVar(value="2.58")
        self.scale_var = tk.StringVar(value=SCALE_AUTO)
        self.status_var = tk.StringVar(value="Ready")

        # Deformation analysis variables. The workbook column defaults match the
        # standard IBHS deformation export, including the supplied sample file.
        self.deformation_input_var = tk.StringVar()
        self.deformation_output_var = tk.StringVar()
        self.deformation_sheet_var = tk.StringVar()
        self.deformation_name_column_var = tk.StringVar(value="A")
        self.deformation_image_column_var = tk.StringVar(value="D")
        self.deformation_scale_column_var = tk.StringVar(value="G")
        self.deformation_legend_column_var = tk.StringVar(value="H")
        self.deformation_header_rows_var = tk.StringVar(
            value=str(_deformation_default("header_search_rows"))
        )
        self.deformation_depth_min_var = tk.StringVar()
        self.deformation_depth_max_var = tk.StringVar()
        self.deformation_dent_threshold_var = tk.StringVar(
            value=str(_deformation_default("dent_depth_threshold_mm"))
        )
        self.deformation_baseline_var = tk.StringVar(
            value=str(_deformation_default("baseline_depth_mm"))
        )
        self.deformation_min_area_var = tk.StringVar(
            value=str(_deformation_default("min_region_area_mm2"))
        )
        self.deformation_min_pixels_var = tk.StringVar(
            value=str(_deformation_default("min_region_pixels"))
        )
        self.deformation_scale_mode_var = tk.StringVar(value=DEFORMATION_SCALE_OCR)
        self.deformation_mm_per_pixel_var = tk.StringVar()
        self.deformation_mm_per_pixel_x_var = tk.StringVar()
        self.deformation_mm_per_pixel_y_var = tk.StringVar()
        self.deformation_max_color_distance_var = tk.StringVar(
            value=str(_deformation_default("max_color_distance"))
        )
        self.deformation_ocr_max_depth_var = tk.StringVar(
            value=str(_deformation_default("ocr_max_abs_depth_mm"))
        )
        self.deformation_legend_clipped_var = tk.BooleanVar(value=False)
        self.deformation_include_edges_var = tk.BooleanVar(value=False)
        self.deformation_edge_margin_var = tk.StringVar(
            value=str(_deformation_default("edge_margin_pixels"))
        )
        self.deformation_tesseract_var = tk.StringVar(
            value=resolve_tesseract_command(
                str(_deformation_default("tesseract_command"))
            )
        )
        self.deformation_status_var = tk.StringVar(value="Ready")
        self.deformation_events = queue.Queue()

        # Track if processing is running
        self.is_processing = False

        # Create UI
        self.create_widgets()

    def create_widgets(self):
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Section.TLabel', font=('Arial', 10, 'bold'))
        style.configure('Hint.TLabel', foreground='#555555')
        style.configure('Accent.TButton', font=('Arial', 10, 'bold'))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.analysis_notebook = ttk.Notebook(self.root)
        self.analysis_notebook.grid(
            row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=8, pady=8
        )

        granule_tab = ttk.Frame(self.analysis_notebook)
        deformation_tab = ttk.Frame(self.analysis_notebook)
        deformation_help_tab = ttk.Frame(self.analysis_notebook)
        self.analysis_notebook.add(granule_tab, text="Granule Loss")
        self.analysis_notebook.add(deformation_tab, text="Deformation Volume")
        self.analysis_notebook.add(
            deformation_help_tab, text="Deformation Options Guide"
        )

        granule_tab.columnconfigure(0, weight=1)
        granule_tab.rowconfigure(0, weight=1)
        main_frame = ttk.Frame(granule_tab, padding="12")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(5, weight=1)

        ttk.Label(main_frame, text="IBHS Granule Loss Analysis", style='Title.TLabel').grid(
            row=0, column=0, sticky=tk.W
        )
        ttk.Label(
            main_frame,
            text="Select a folder of scale-bar images. Cropped and annotated analysis copies are saved beside each source image.",
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

        ttk.Label(settings_frame, text="Scale bar length", style='Section.TLabel').grid(
            row=1, column=0, sticky=tk.W, pady=(8, 0)
        )
        ttk.Combobox(
            settings_frame,
            textvariable=self.scale_var,
            values=SCALE_CHOICES,
            state="readonly",
            width=12,
        ).grid(row=1, column=1, sticky=tk.W, padx=(8, 4), pady=(8, 0))
        ttk.Label(
            settings_frame,
            text=(
                f"Auto reads the mm label printed next to each bar; a {SCALE_TAGS} "
                f"file-name tag overrides it. Set a value here to force it for every image."
            ),
        ).grid(row=1, column=2, columnspan=2, sticky=tk.W, padx=(8, 0), pady=(8, 0))

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
            "AreaSum_All_mm2",
            "GL_Rating",
            "CombinedGL_Rating",
            "ScaleBar_mm",
            "Scale_Source",
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

        self.create_deformation_widgets(deformation_tab)
        self.create_deformation_help_widgets(deformation_help_tab)

    def create_deformation_help_widgets(self, parent):
        """Explain every deformation option without crowding the analysis form."""
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        help_text = scrolledtext.ScrolledText(
            parent,
            wrap=tk.WORD,
            padx=24,
            pady=20,
            font=('Arial', 11),
            spacing1=2,
            spacing3=8,
        )
        help_text.grid(
            row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=12, pady=12
        )
        help_text.tag_configure("title", font=('Arial', 18, 'bold'), spacing3=10)
        help_text.tag_configure("heading", font=('Arial', 13, 'bold'), spacing1=12)
        help_text.tag_configure("option", font=('Arial', 11, 'bold'))
        help_text.tag_configure("note", foreground="#555555")

        def add(text, tag=None):
            help_text.insert(tk.END, text, tag)

        add("Deformation Options Guide\n", "title")
        add(
            "The deformation analyzer reads false-color height-difference images "
            "embedded in an Excel workbook, converts their colors to physical depth, "
            "finds dent regions, and calculates area and volume. It never edits the "
            "source workbook.\n"
        )

        add("Quick start for the supplied sample\n", "heading")
        add("1. Select ", None)
        add("26H01_107_Z_P1_Deformation.xlsx", "option")
        add(" as the input file.\n")
        add("2. Choose a results folder.\n")
        add("3. Keep the column defaults: ")
        add("A", "option")
        add(" measurement name, ")
        add("D", "option")
        add(" false-color image, ")
        add("G", "option")
        add(" scaled companion, and ")
        add("H", "option")
        add(" color legend.\n")
        add("4. Keep ")
        add("OCR from scale images", "option")
        add(" selected and leave Depth minimum/maximum blank.\n")
        add("5. Click ")
        add("Analyze Workbook", "option")
        add(". The Summary tab displays the CSV results when processing finishes.\n")

        add("Workbook options\n", "heading")
        add("Input Excel file — ", "option")
        add("The .xlsx or .xlsm workbook containing the embedded images.\n")
        add("Results folder — ", "option")
        add(
            "Where extracted originals, annotated images, and the dent-volume CSV "
            "will be stored. If blank, an analysis_output folder is created beside "
            "the workbook.\n"
        )
        add("Worksheet — ", "option")
        add(
            "The exact sheet name to analyze. Leave it blank to use the workbook's "
            "active sheet. The sample's active sheet is Compare.\n"
        )

        add("Workbook column mapping\n", "heading")
        add(
            "Each column can be identified by an Excel letter such as D or by exact "
            "header text found near the top of the sheet.\n",
            "note",
        )
        add("Measurement name — ", "option")
        add("Names each analyzed row and its output files.\n")
        add("False-color image — ", "option")
        add("Contains the color-mapped height-difference image used to find dents.\n")
        add("Scaled companion — ", "option")
        add(
            "Contains the same measurement with labeled X/Y physical extents. OCR "
            "uses it to calculate millimeters per pixel.\n"
        )
        add("Color legend — ", "option")
        add(
            "Contains the vertical color bar that maps image colors to depth. It is "
            "optional only when both Depth minimum and Depth maximum are supplied.\n"
        )
        add("Header search rows — ", "option")
        add(
            "How many rows from the top are searched when header text is used instead "
            "of a column letter. It does not limit which data rows are analyzed.\n"
        )

        add("Depth calibration\n", "heading")
        add("Depth minimum / maximum — ", "option")
        add(
            "Optional explicit endpoints, in millimeters, for the color scale. Leave "
            "both blank to OCR the endpoint labels in every row's legend. Enter both "
            "or neither; the minimum must be less than the maximum. If no legend "
            "column is provided, both values are required and a standard jet palette "
            "is used.\n"
        )
        add("Dent threshold — ", "option")
        add(
            "Pixels at or below this depth become candidates for a dent. The default "
            "is -0.04 mm; making it more negative keeps only deeper depressions.\n"
        )
        add("Baseline depth — ", "option")
        add(
            "The reference surface used when integrating dent volume. It must be "
            "greater than the dent threshold. The default is 0 mm.\n"
        )
        add("Max color distance — ", "option")
        add(
            "Maximum RGB difference allowed when matching an image pixel to the "
            "legend palette. Lower values reject more off-palette pixels; higher "
            "values accept looser color matches.\n"
        )
        add("OCR safety limit — ", "option")
        add(
            "Rejects legend endpoint readings whose absolute depth exceeds this many "
            "millimeters. It protects against obviously incorrect OCR.\n"
        )

        add("Physical scale\n", "heading")
        add("OCR from scale images — ", "option")
        add(
            "Recommended when the scaled companion plots have readable axis labels. "
            "Rows that cannot be read use the median scale from successful rows.\n"
        )
        add("Fixed square mm/pixel — ", "option")
        add(
            "Skips scale OCR and uses one value for both X and Y. Choose this only "
            "when the pixels are square and the physical resolution is known.\n"
        )
        add("Fixed X/Y mm/pixel — ", "option")
        add(
            "Skips scale OCR and uses separate horizontal and vertical resolutions. "
            "Use this for non-square pixels.\n"
        )
        add("Tesseract command — ", "option")
        add(
            "The executable name or full path for Tesseract OCR. The default is "
            "tesseract. It must be installed when scale extents or legend endpoints "
            "are read automatically.\n"
        )

        add("Region filtering\n", "heading")
        add("Minimum area — ", "option")
        add(
            "A connected candidate region must cover at least this physical area, in "
            "mm². Increase it to ignore small depressions.\n"
        )
        add("Minimum pixels — ", "option")
        add(
            "A second size safeguard in raw pixels. A region must meet both the "
            "pixel requirement and the physical-area requirement.\n"
        )
        add("Edge margin — ", "option")
        add(
            "Regions within this many pixels of an image boundary count as edge "
            "regions. The default behavior excludes them.\n"
        )
        add("Include regions touching edges — ", "option")
        add(
            "Keeps edge regions instead of excluding them. Enable this only when "
            "boundary depressions are valid measurements rather than cropped shapes.\n"
        )
        add("Legend minimum is clipped — ", "option")
        add(
            "Use this when the true dent depth may extend below the displayed legend "
            "minimum. Any affected volume is then marked as a lower bound; this option "
            "does not change the volume calculation itself.\n"
        )

        add("Generated results\n", "heading")
        add(
            "For every non-black source row, the analyzer stores the embedded original "
            "image and an annotated PNG showing retained dents and volumes. It also "
            "writes one CSV containing dent counts, per-dent values, total area and "
            "volume, scale and legend sources, depth statistics, warnings, and output "
            "file names. Black placeholder rows are ignored.\n"
        )

        help_text.configure(state=tk.DISABLED)

    def create_deformation_widgets(self, parent):
        """Build the workbook-based deformation analysis tab."""
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        main_frame = ttk.Frame(parent, padding="12")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(6, weight=1)

        ttk.Label(
            main_frame, text="IBHS Deformation Volume Analysis", style='Title.TLabel'
        ).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(
            main_frame,
            text=(
                "Analyze height-difference images embedded in an Excel workbook. "
                "The source workbook remains unchanged; CSV data and annotated images "
                "are written to the results folder."
            ),
        ).grid(row=1, column=0, sticky=tk.W, pady=(2, 10))

        workbook_frame = ttk.LabelFrame(main_frame, text="Workbook", padding="10")
        workbook_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        workbook_frame.columnconfigure(1, weight=1)

        ttk.Label(workbook_frame, text="Input Excel file", style='Section.TLabel').grid(
            row=0, column=0, sticky=tk.W, pady=3
        )
        ttk.Entry(
            workbook_frame, textvariable=self.deformation_input_var, width=70
        ).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=8, pady=3)
        ttk.Button(
            workbook_frame, text="Browse...", command=self.browse_deformation_workbook
        ).grid(row=0, column=2, pady=3)

        ttk.Label(workbook_frame, text="Results folder", style='Section.TLabel').grid(
            row=1, column=0, sticky=tk.W, pady=3
        )
        ttk.Entry(
            workbook_frame, textvariable=self.deformation_output_var, width=70
        ).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=8, pady=3)
        ttk.Button(
            workbook_frame, text="Browse...", command=self.browse_deformation_output
        ).grid(row=1, column=2, pady=3)

        ttk.Label(workbook_frame, text="Worksheet").grid(
            row=2, column=0, sticky=tk.W, pady=3
        )
        ttk.Entry(
            workbook_frame, textvariable=self.deformation_sheet_var, width=24
        ).grid(row=2, column=1, sticky=tk.W, padx=8, pady=3)
        ttk.Label(workbook_frame, text="Leave blank to use the active worksheet.").grid(
            row=2, column=1, sticky=tk.W, padx=(210, 0), pady=3
        )

        columns_frame = ttk.LabelFrame(
            main_frame, text="Workbook columns (header text or Excel letter)", padding="10"
        )
        columns_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        for column in range(5):
            columns_frame.columnconfigure(column, weight=1)
        column_fields = (
            ("Measurement name", self.deformation_name_column_var),
            ("False-color image", self.deformation_image_column_var),
            ("Scaled companion", self.deformation_scale_column_var),
            ("Color legend", self.deformation_legend_column_var),
            ("Header search rows", self.deformation_header_rows_var),
        )
        for column, (label, variable) in enumerate(column_fields):
            ttk.Label(columns_frame, text=label).grid(
                row=0, column=column, sticky=tk.W, padx=(0, 8)
            )
            ttk.Entry(columns_frame, textvariable=variable, width=19).grid(
                row=1, column=column, sticky=(tk.W, tk.E), padx=(0, 8), pady=(3, 0)
            )

        settings_frame = ttk.Frame(main_frame)
        settings_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        for column in range(3):
            settings_frame.columnconfigure(column, weight=1, uniform="deformation")

        depth_frame = ttk.LabelFrame(settings_frame, text="Depth calibration", padding="10")
        depth_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 4))
        depth_frame.columnconfigure(1, weight=1)
        depth_fields = (
            ("Depth minimum (mm)", self.deformation_depth_min_var),
            ("Depth maximum (mm)", self.deformation_depth_max_var),
            ("Dent threshold (mm)", self.deformation_dent_threshold_var),
            ("Baseline depth (mm)", self.deformation_baseline_var),
            ("Max color distance", self.deformation_max_color_distance_var),
            ("OCR safety limit (mm)", self.deformation_ocr_max_depth_var),
        )
        for row, (label, variable) in enumerate(depth_fields):
            ttk.Label(depth_frame, text=label).grid(row=row, column=0, sticky=tk.W, pady=2)
            ttk.Entry(depth_frame, textvariable=variable, width=11).grid(
                row=row, column=1, sticky=tk.E, padx=(8, 0), pady=2
            )
        ttk.Label(
            depth_frame,
            text="Leave min/max blank to OCR each row legend.",
            style='Hint.TLabel',
        ).grid(row=len(depth_fields), column=0, columnspan=2, sticky=tk.W, pady=(5, 0))

        scale_frame = ttk.LabelFrame(settings_frame, text="Physical scale", padding="10")
        scale_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=4)
        scale_frame.columnconfigure(1, weight=1)
        ttk.Label(scale_frame, text="Calibration mode").grid(row=0, column=0, sticky=tk.W, pady=2)
        scale_mode = ttk.Combobox(
            scale_frame,
            textvariable=self.deformation_scale_mode_var,
            values=DEFORMATION_SCALE_MODES,
            state="readonly",
            width=22,
        )
        scale_mode.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(8, 0), pady=2)
        scale_mode.bind("<<ComboboxSelected>>", self._update_deformation_scale_fields)

        ttk.Label(scale_frame, text="Square mm/pixel").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.deformation_square_entry = ttk.Entry(
            scale_frame, textvariable=self.deformation_mm_per_pixel_var, width=11
        )
        self.deformation_square_entry.grid(row=1, column=1, sticky=tk.E, padx=(8, 0), pady=2)

        ttk.Label(scale_frame, text="X mm/pixel").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.deformation_x_entry = ttk.Entry(
            scale_frame, textvariable=self.deformation_mm_per_pixel_x_var, width=11
        )
        self.deformation_x_entry.grid(row=2, column=1, sticky=tk.E, padx=(8, 0), pady=2)

        ttk.Label(scale_frame, text="Y mm/pixel").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.deformation_y_entry = ttk.Entry(
            scale_frame, textvariable=self.deformation_mm_per_pixel_y_var, width=11
        )
        self.deformation_y_entry.grid(row=3, column=1, sticky=tk.E, padx=(8, 0), pady=2)

        ttk.Label(scale_frame, text="Tesseract command").grid(row=4, column=0, sticky=tk.W, pady=2)
        ttk.Entry(
            scale_frame, textvariable=self.deformation_tesseract_var, width=18
        ).grid(row=4, column=1, sticky=(tk.W, tk.E), padx=(8, 0), pady=2)
        ttk.Label(
            scale_frame,
            text="OCR mode reads the companion plot extents.",
            style='Hint.TLabel',
        ).grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))

        region_frame = ttk.LabelFrame(settings_frame, text="Region filtering", padding="10")
        region_frame.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(4, 0))
        region_frame.columnconfigure(1, weight=1)
        region_fields = (
            ("Minimum area (mm²)", self.deformation_min_area_var),
            ("Minimum pixels", self.deformation_min_pixels_var),
            ("Edge margin (pixels)", self.deformation_edge_margin_var),
        )
        for row, (label, variable) in enumerate(region_fields):
            ttk.Label(region_frame, text=label).grid(row=row, column=0, sticky=tk.W, pady=2)
            ttk.Entry(region_frame, textvariable=variable, width=11).grid(
                row=row, column=1, sticky=tk.E, padx=(8, 0), pady=2
            )
        ttk.Checkbutton(
            region_frame,
            text="Include regions touching edges",
            variable=self.deformation_include_edges_var,
        ).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(5, 2))
        ttk.Checkbutton(
            region_frame,
            text="Legend minimum is clipped",
            variable=self.deformation_legend_clipped_var,
        ).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=2)
        ttk.Label(
            region_frame,
            text="Clipped legends mark affected volumes as lower bounds.",
            style='Hint.TLabel',
        ).grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))

        action_frame = ttk.Frame(main_frame)
        action_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        action_frame.columnconfigure(1, weight=1)
        self.deformation_run_button = ttk.Button(
            action_frame,
            text="Analyze Workbook",
            command=self.run_deformation,
            style='Accent.TButton',
        )
        self.deformation_run_button.grid(row=0, column=0, sticky=tk.W)
        self.deformation_progress = ttk.Progressbar(action_frame, mode='indeterminate')
        self.deformation_progress.grid(
            row=0, column=1, sticky=(tk.W, tk.E), padx=12
        )
        ttk.Label(action_frame, textvariable=self.deformation_status_var).grid(
            row=0, column=2, sticky=tk.E
        )

        self.deformation_notebook = ttk.Notebook(main_frame)
        self.deformation_notebook.grid(
            row=6, column=0, sticky=(tk.W, tk.E, tk.N, tk.S)
        )
        deformation_summary = ttk.Frame(self.deformation_notebook, padding="5")
        deformation_log = ttk.Frame(self.deformation_notebook)
        self.deformation_notebook.add(deformation_summary, text="Summary")
        self.deformation_notebook.add(deformation_log, text="Logs")

        deformation_summary.columnconfigure(0, weight=1)
        deformation_summary.rowconfigure(0, weight=1)
        deformation_columns = (
            "measurement_data_name",
            "dent_count",
            "total_volume_mm3",
            "total_area_mm2",
            "minimum_depth_mm",
            "mean_dent_depth_mm",
            "mm_per_pixel_x",
            "mm_per_pixel_y",
            "scale_source",
            "legend_source",
            "volume_is_lower_bound",
            "notes",
        )
        self.deformation_summary_tree = ttk.Treeview(
            deformation_summary,
            columns=deformation_columns,
            show="headings",
            height=5,
        )
        for column in deformation_columns:
            self.deformation_summary_tree.heading(column, text=column)
            if column == "measurement_data_name":
                width = 150
            elif column == "notes":
                width = 180
            else:
                width = 80
            self.deformation_summary_tree.column(
                column, width=width, minwidth=70, anchor=tk.W
            )
        self.deformation_summary_tree.grid(
            row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S)
        )
        deformation_v_scroll = ttk.Scrollbar(
            deformation_summary,
            orient=tk.VERTICAL,
            command=self.deformation_summary_tree.yview,
        )
        deformation_v_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        deformation_h_scroll = ttk.Scrollbar(
            deformation_summary,
            orient=tk.HORIZONTAL,
            command=self.deformation_summary_tree.xview,
        )
        deformation_h_scroll.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.deformation_summary_tree.configure(
            yscrollcommand=deformation_v_scroll.set,
            xscrollcommand=deformation_h_scroll.set,
        )

        self.deformation_log_text = scrolledtext.ScrolledText(
            deformation_log, width=80, height=6, wrap=tk.WORD, font=('Courier', 9)
        )
        self.deformation_log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self._update_deformation_scale_fields()

    def browse_deformation_workbook(self):
        filename = filedialog.askopenfilename(
            title="Select Deformation Workbook",
            filetypes=(
                ("Excel workbooks", "*.xlsx *.xlsm"),
                ("All files", "*.*"),
            ),
        )
        if filename:
            self.deformation_input_var.set(filename)
            if not self.deformation_output_var.get().strip():
                self.deformation_output_var.set(
                    str(Path(filename).parent / "analysis_output")
                )

    def browse_deformation_output(self):
        folder = filedialog.askdirectory(title="Select Deformation Results Folder")
        if folder:
            self.deformation_output_var.set(folder)

    def _update_deformation_scale_fields(self, _event=None):
        mode = self.deformation_scale_mode_var.get()
        self.deformation_square_entry.configure(
            state="normal" if mode == DEFORMATION_SCALE_SQUARE else "disabled"
        )
        xy_state = "normal" if mode == DEFORMATION_SCALE_XY else "disabled"
        self.deformation_x_entry.configure(state=xy_state)
        self.deformation_y_entry.configure(state=xy_state)

    @staticmethod
    def _number(value, label, number_type=float, optional=False):
        text = value.strip()
        if optional and not text:
            return None
        try:
            return number_type(text)
        except (TypeError, ValueError) as exc:
            qualifier = "integer" if number_type is int else "number"
            raise ValueError(f"{label} must be a valid {qualifier}.") from exc

    def _build_deformation_args(self):
        input_text = self.deformation_input_var.get().strip()
        if not input_text:
            raise ValueError("Select an input Excel workbook.")
        input_path = Path(input_text)
        if not input_path.is_file():
            raise ValueError("The input Excel workbook does not exist.")

        output_text = self.deformation_output_var.get().strip()
        if not output_text:
            output_text = str(input_path.parent / "analysis_output")
            self.deformation_output_var.set(output_text)
        output_path = Path(output_text)
        if output_path.exists() and not output_path.is_dir():
            raise ValueError("The results path exists and is not a folder.")

        selectors = {
            "name_column": self.deformation_name_column_var.get().strip(),
            "image_column": self.deformation_image_column_var.get().strip(),
            "scale_column": self.deformation_scale_column_var.get().strip(),
        }
        for name, value in selectors.items():
            if not value:
                label = name.replace("_", " ").capitalize()
                raise ValueError(f"{label} is required.")

        depth_min = self._number(
            self.deformation_depth_min_var.get(), "Depth minimum", optional=True
        )
        depth_max = self._number(
            self.deformation_depth_max_var.get(), "Depth maximum", optional=True
        )
        legend_column = self.deformation_legend_column_var.get().strip() or None
        if legend_column is None and depth_min is None and depth_max is None:
            raise ValueError(
                "Enter a color legend column, or supply both depth minimum and maximum."
            )

        mm_per_pixel = None
        mm_per_pixel_x = None
        mm_per_pixel_y = None
        scale_mode = self.deformation_scale_mode_var.get()
        if scale_mode == DEFORMATION_SCALE_SQUARE:
            mm_per_pixel = self._number(
                self.deformation_mm_per_pixel_var.get(), "Square mm/pixel"
            )
        elif scale_mode == DEFORMATION_SCALE_XY:
            mm_per_pixel_x = self._number(
                self.deformation_mm_per_pixel_x_var.get(), "X mm/pixel"
            )
            mm_per_pixel_y = self._number(
                self.deformation_mm_per_pixel_y_var.get(), "Y mm/pixel"
            )
        elif scale_mode != DEFORMATION_SCALE_OCR:
            raise ValueError("Select a valid physical-scale calibration mode.")

        header_search_rows = self._number(
            self.deformation_header_rows_var.get(), "Header search rows", int
        )
        if header_search_rows < 1:
            raise ValueError("Header search rows must be at least 1.")

        tesseract_command = self.deformation_tesseract_var.get().strip()
        if not tesseract_command:
            raise ValueError("Tesseract command cannot be blank.")

        args = argparse.Namespace(
            input_xlsx=input_path,
            sheet=self.deformation_sheet_var.get().strip() or None,
            name_column=selectors["name_column"],
            image_column=selectors["image_column"],
            scale_column=selectors["scale_column"],
            legend_column=legend_column,
            header_search_rows=header_search_rows,
            output_dir=output_path,
            depth_min_mm=depth_min,
            depth_max_mm=depth_max,
            dent_depth_threshold_mm=self._number(
                self.deformation_dent_threshold_var.get(), "Dent threshold"
            ),
            baseline_depth_mm=self._number(
                self.deformation_baseline_var.get(), "Baseline depth"
            ),
            min_region_area_mm2=self._number(
                self.deformation_min_area_var.get(), "Minimum region area"
            ),
            min_region_pixels=self._number(
                self.deformation_min_pixels_var.get(), "Minimum region pixels", int
            ),
            mm_per_pixel=mm_per_pixel,
            mm_per_pixel_x=mm_per_pixel_x,
            mm_per_pixel_y=mm_per_pixel_y,
            max_color_distance=self._number(
                self.deformation_max_color_distance_var.get(), "Max color distance"
            ),
            ocr_max_abs_depth_mm=self._number(
                self.deformation_ocr_max_depth_var.get(), "OCR safety limit"
            ),
            legend_is_clipped=bool(self.deformation_legend_clipped_var.get()),
            include_edge_regions=bool(self.deformation_include_edges_var.get()),
            edge_margin_pixels=self._number(
                self.deformation_edge_margin_var.get(), "Edge margin", int
            ),
            tesseract_command=tesseract_command,
        )
        validate_deformation_args(args)
        return args

    def deformation_log_message(self, message):
        self.deformation_events.put(("log", str(message)))

    def _append_deformation_log(self, message):
        self.deformation_log_text.insert(tk.END, message + "\n")
        self.deformation_log_text.see(tk.END)

    def run_deformation(self):
        """Validate the form and analyze the workbook on a worker thread."""
        if self.is_processing:
            messagebox.showwarning("Warning", "An analysis is already running.")
            return
        try:
            args = self._build_deformation_args()
        except ValueError as exc:
            messagebox.showerror("Invalid Deformation Settings", str(exc))
            return

        self.deformation_log_text.delete(1.0, tk.END)
        for item in self.deformation_summary_tree.get_children():
            self.deformation_summary_tree.delete(item)

        self.is_processing = True
        self.deformation_run_button.configure(state="disabled")
        self.run_button.configure(state="disabled")
        self.deformation_status_var.set("Running...")
        self.deformation_progress.start()
        self.deformation_notebook.select(1)
        self.deformation_events = queue.Queue()

        thread = threading.Thread(
            target=self._run_deformation_thread, args=(args,), daemon=True
        )
        thread.start()
        self.root.after(50, self._poll_deformation_events)

    def _run_deformation_thread(self, args):
        writer = _LineLogWriter(self.deformation_log_message)
        try:
            self.deformation_log_message("=" * 70)
            self.deformation_log_message("Starting Deformation Volume Analysis...")
            self.deformation_log_message("=" * 70)
            self.deformation_log_message(f"Workbook: {args.input_xlsx}")
            self.deformation_log_message(
                f"Worksheet: {args.sheet or 'active worksheet'}"
            )
            self.deformation_log_message(f"Results folder: {args.output_dir}\n")

            with contextlib.redirect_stdout(writer):
                csv_path = run_deformation_analysis(args)
            writer.flush()

            with csv_path.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.deformation_events.put(("summary", rows))

            self.deformation_log_message("\n" + "=" * 70)
            self.deformation_log_message("Analysis completed successfully.")
            self.deformation_log_message(f"Summary CSV: {csv_path}")
            self.deformation_log_message("=" * 70)
            self.deformation_events.put(
                (
                    "info",
                    "Deformation Analysis Complete",
                    f"Analyzed {len(rows)} workbook row(s).\n\n"
                    f"Results saved to:\n{args.output_dir}\n\n"
                    f"Summary CSV:\n{csv_path.name}",
                )
            )
        except Exception as exc:
            writer.flush()
            error_message = f"Error during deformation analysis: {exc}"
            self.deformation_log_message("\n" + error_message)
            self.deformation_events.put(
                ("error", "Deformation Analysis Error", error_message)
            )
        finally:
            self.deformation_events.put(("finished",))

    def _poll_deformation_events(self):
        finished = False
        while True:
            try:
                event = self.deformation_events.get_nowait()
            except queue.Empty:
                break
            event_type = event[0]
            if event_type == "log":
                self._append_deformation_log(event[1])
            elif event_type == "summary":
                self._display_deformation_summary(event[1])
            elif event_type == "info":
                messagebox.showinfo(event[1], event[2])
            elif event_type == "error":
                messagebox.showerror(event[1], event[2])
            elif event_type == "finished":
                self._finish_deformation_processing()
                finished = True
        if not finished and self.is_processing:
            self.root.after(50, self._poll_deformation_events)

    def _display_deformation_summary(self, rows):
        for item in self.deformation_summary_tree.get_children():
            self.deformation_summary_tree.delete(item)
        columns = self.deformation_summary_tree["columns"]
        for row in rows:
            self.deformation_summary_tree.insert(
                "", tk.END, values=[row.get(column, "") for column in columns]
            )
        self.deformation_notebook.select(0)

    def _finish_deformation_processing(self):
        self.deformation_progress.stop()
        self.deformation_run_button.configure(state="normal")
        self.run_button.configure(state="normal")
        self.deformation_status_var.set("Ready")
        self.is_processing = False

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

    def selected_scale_mm(self):
        """Return the forced scale-bar length in mm, or None when auto-detecting."""
        choice = self.scale_var.get()
        if choice == SCALE_AUTO:
            return None
        return float(choice.split()[0])

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
        self.deformation_run_button.config(state='disabled')
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
            forced_scale_mm = self.selected_scale_mm()

            self.log_message(f"\nInput Folder: {input_folder}")
            self.log_message(f"Output Folder: {output_folder}")
            self.log_message(f"IGL/PGL Threshold: {threshold} mm2")
            self.log_message(
                f"Scale bar: {'forced to %g mm for every image' % forced_scale_mm}"
                if forced_scale_mm
                else "Scale bar: auto-detected per image (file-name tag, then printed label)"
            )
            self.log_message("\nCropped images will be saved beside each source as *_cropped.*")
            self.log_message("Annotated images with per-spot areas will be saved as *_annotated.*\n")

            # Run the analysis
            summary_df, fig = process_granule_loss(
                input_folder=input_folder,
                output_folder=output_folder,
                igl_cutoff_mm2=threshold,
                log_callback=self.log_message,
                forced_scale_mm=forced_scale_mm,
            )

            self.root.after(0, self._display_summary, summary_df)
            self.root.after(0, self._display_plot, fig)

            self.log_message("\n" + "=" * 70)
            self.log_message("Analysis completed successfully!")
            self.log_message("=" * 70)

            # Show success message, calling out any image whose scale is a guess
            message = f"Analysis completed!\n\nResults saved to:\n{output_folder}"
            if "Scale_Verified" in summary_df.columns:
                unverified = summary_df.loc[~summary_df["Scale_Verified"].astype(bool), "Impact"]
                if len(unverified):
                    message += (
                        f"\n\nWARNING: the scale bar length could not be verified for "
                        f"{len(unverified)} of {len(summary_df)} image(s):\n"
                        f"{', '.join(map(str, unverified[:8]))}"
                        f"{' ...' if len(unverified) > 8 else ''}\n\n"
                        "Their areas may be wrong by 4x. Set 'Scale bar length' explicitly "
                        f"and re-run, or tag those file names with {SCALE_TAGS}."
                    )
            self.root.after(0, messagebox.showinfo, "Success", message)

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
        self.deformation_run_button.config(state='normal')
        self.status_var.set("Ready")
        self.is_processing = False


def main():
    root = tk.Tk()
    app = GranuleLossApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
