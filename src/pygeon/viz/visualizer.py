import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

try:
    import pyvista as pv

    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False


def _check_latex_available() -> bool:
    """Check if LaTeX is available on the system."""
    return shutil.which("latex") is not None


class Visualizer:
    """
    A flexible visualization class for PVD time-series mesh files using PyVista.

    Supports both Jupyter notebook and standalone usage with various
    visualization modes for scalar and vector fields. Handles PVD files
    containing multiple grids of different dimensions (1D, 2D, 3D) at
    different time steps.

    Requires PyVista to be installed.

    Example:
        Basic usage::

            vis = Visualizer("results")
            vis.scalar_field("pressure", cmap="viridis")
            vis.show(screenshot="pressure.png")
    """

    def __init__(
        self, file_name: str | Path, folder_name: str | Path = "", time_step: int = 0
    ) -> None:
        """
        Initialize the Visualizer.

        Args:
            file_name (str | Path): Name or path to the PVD file
                (extension .pvd can be omitted).
            folder_name (str | Path): Optional folder path. If provided,
                will be combined with file_name.
            time_step (int): Time step index for PVD files.
                Default 0 (first time step).
        """
        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista is required for visualization.")

        # Convert to Path objects if strings
        file_name = Path(file_name)

        # Construct full path as union of folder_name and file_name
        if folder_name != "":
            folder_name = Path(folder_name)
            file_name = folder_name / file_name

        # Add .pvd extension if not present
        if file_name.suffix.lower() != ".pvd":
            file_name = file_name.with_suffix(".pvd")

        self.file_path: Path = file_name
        self.time_step: int = time_step

        # Configure PyVista theme
        pv.global_theme.font.label_size = 14
        pv.global_theme.font.title_size = 16

        # Enable LaTeX rendering if available
        if _check_latex_available():
            pv.global_theme.font.family = "times"
            # LaTeX is available, PyVista can use it for text rendering
        else:
            pv.global_theme.font.family = "arial"

        self.plotter: "pv.Plotter" = pv.Plotter()

        # Load the PVD file
        self.meshes: dict[str, "pv.DataSet"] = {}
        self.active_mesh_name: str = ""
        self.time_values: list[float] = []
        self.current_time: float = 0.0
        self.actors: list[Any] = []
        self._load_pvd(file_name, time_step)

    def _load_pvd(self, file_path: Path, time_step: int = 0) -> None:
        """
        Load a PVD file containing time-series data.

        The mesh geometry remains constant across timesteps; only the scalar
        and vector fields change. Meshes are indexed by dimension extracted from
        filenames (e.g., sol_2_000000.vtu indicates dimension 2).
        All grids present at the first timestep are assumed to be present at all
        timesteps.

        Args:
            file_path (Path): Path to the PVD file.
            time_step (int): Time step index to load (0-based).
        """
        # Parse PVD XML file to extract timesteps and files
        tree = ET.parse(str(file_path))
        root = tree.getroot()

        # Extract all DataSet entries from the Collection
        datasets = root.findall(".//DataSet")

        # Parse timesteps and file paths
        time_data = []
        for dataset in datasets:
            timestep = float(dataset.get("timestep", 0))
            file_ref = dataset.get("file")
            if file_ref:
                time_data.append((timestep, file_ref))

        # Sort by timestep
        time_data.sort(key=lambda x: x[0])

        # Load meshes from first timestep
        # Extract unique dimensions from the first timestep files
        first_timestep = time_data[0][0]
        base_dir = file_path.parent

        # Collect all unique dimensions from first timestep files
        file_to_dim = {}  # Map from filename to dimension

        for timestep, file_ref in time_data:
            if timestep == first_timestep:
                # Extract dimension from filename (e.g., "sol_2_000000.vtu" -> dim 2)
                dim = self._extract_dimension_from_filename(file_ref)
                if dim is not None:
                    file_to_dim[file_ref] = dim

        # Load mesh for each dimension from first timestep
        for timestep, file_ref in time_data:
            if timestep == first_timestep:
                dim = file_to_dim.get(file_ref)
                if dim is not None:
                    mesh = pv.read(str(base_dir / file_ref))
                    self.meshes[f"Mesh_dim{dim}"] = mesh

        # Store time information
        self.time_values = [t[0] for t in time_data]
        self.current_time = time_data[time_step][0]

        # Set the first mesh as active
        self.active_mesh_name = list(self.meshes.keys())[0]

    @staticmethod
    def _extract_dimension_from_filename(filename: str) -> int | None:
        """
        Extract spatial dimension from filename.

        Assumes format like 'sol_2_000000.vtu' where _2_ indicates dimension 2.
        The dimension is the digit immediately before the timestep number.

        Args:
            filename (str): The VTU filename.

        Returns:
            int | None: The dimension (0, 1, 2, or 3) or None if not found.
        """
        # Remove .vtu extension and split by underscore
        parts = filename.replace(".vtu", "").split("_")
        
        # The timestep is typically the last part (all digits)
        # The dimension is the part immediately before it
        if len(parts) >= 2:
            # Check if the last part is a timestep (all digits)
            if parts[-1].isdigit():
                # Get the part before the timestep
                dim_part = parts[-2]
                if dim_part.isdigit():
                    dim = int(dim_part)
                    if dim in [0, 1, 2, 3]:
                        return dim
        return None

    @property
    def mesh(self) -> "pv.DataSet":
        """Get the currently active mesh."""
        return self.meshes[self.active_mesh_name]

    def load_time_step(self, time_index: int) -> None:
        """
        Load a specific time step from the PVD file.

        Args:
            time_index (int): Index of the time step (0-based).
        """
        # Clear and reload at new time step
        self.clear()
        # Then reset mesh data (time_values and current_time are set by _load_pvd)
        self.meshes = {}
        # Finally reload at new time step
        self._load_pvd(self.file_path, time_index)

    def _default_bar_args(self, field_name: str) -> dict[str, Any]:
        """Return centered right-side scalar bar args based on mesh dimension."""

        # Extract dimension from active mesh name (e.g., "Mesh_dim2" -> 2)
        if "dim" in self.active_mesh_name:
            dim = int(self.active_mesh_name.split("dim")[-1])
        else:
            dim = 3

        # Make bar size proportional to dimension and center vertically
        # Height grows with dimension but is clamped for readability
        height = 0.35 + 0.1 * max(1, min(dim, 3))  # 1D->0.45, 2D->0.55, 3D->0.65
        height = min(max(height, 0.35), 0.7)
        position_y = (1.0 - height) / 2.0  # vertical centering

        return {
            "title": field_name,
            "vertical": True,
            "title_font_size": 18,
            "width": 0.08,
            "height": height,
            "position_x": 0.88,
            "position_y": position_y,
        }

    def vector_field(self, field_name: str, scaling_factor: float = 1.0) -> None:
        """
        Visualize a vector field using arrows.

        Args:
            field_name (str): Name of the vector field in the mesh.
            scaling_factor (float): Scaling factor for arrows. Default 1.0.
        """
        arrows = self.mesh.glyph(
            orient=field_name, scale=field_name, factor=scaling_factor, absolute=False
        )
        actor = self.plotter.add_mesh(arrows, color="gray", scalars=None, cmap=None)
        self.actors.append(actor)

    def scalar_field(
        self, field_name: str, field_label: str = None, **kwargs: Any
    ) -> None:
        """
        Visualize a scalar field with color mapping.

        Args:
            field_name (str): Name of the scalar field in the mesh.
            **kwargs: Additional options:

                - cmap (str): Colormap name. Default "rainbow".
                - show_edges (bool): Show mesh edges. Default True.
                - edge_color (str): Color of edges. Default "gray".
                - line_width (float): Width of edge lines. Default 1.0.
                - scalar_bar_args (dict): Scalar bar configuration.
        """
        if field_label is None:
            field_label = field_name

        cmap = kwargs.get("cmap", "rainbow")
        show_edges = kwargs.get("show_edges", True)
        edge_color = kwargs.get("edge_color", "gray")
        line_width = kwargs.get("line_width", 1.0)
        scalar_bar_args = kwargs.get(
            "scalar_bar_args", self._default_bar_args(field_label)
        )

        # Show scalar field without edges to avoid triangulation visibility
        actor = self.plotter.add_mesh(
            self.mesh,
            scalars=field_name,
            cmap=cmap,
            show_edges=False,
            scalar_bar_args=scalar_bar_args,
        )
        self.actors.append(actor)

        # Add cell edges separately if requested
        if show_edges:
            edges = self.mesh.extract_all_edges()
            edge_actor = self.plotter.add_mesh(
                edges,
                color=edge_color,
                line_width=line_width,
                style="wireframe",
            )
            self.actors.append(edge_actor)

    def contour(self, field_name: str, isosurfaces: int = 10) -> None:
        """
        Create contour surfaces of a scalar field.

        Args:
            field_name (str): Name of the scalar field.
            isosurfaces (int): Number of isosurfaces. Default 10.
        """
        # Check if field is cell data, convert to point data if needed
        mesh = self.mesh
        if field_name in mesh.cell_data:
            mesh = mesh.cell_data_to_point_data()

        contours = mesh.contour(isosurfaces=isosurfaces, scalars=field_name)
        actor = self.plotter.add_mesh(contours, color="black", line_width=2.0)
        self.actors.append(actor)

    def show_mesh(self, **kwargs: Any) -> None:
        """
        Visualize the mesh grid without any field data.

        Args:
            **kwargs: Additional options:
                color (str): Mesh color. Default "white".
                show_edges (bool): Show mesh edges. Default True.
                edge_color (str): Color of edges. Default "black".
                opacity (float): Mesh opacity. Default 1.0.
                line_width (float): Width of edge lines. Default 1.0.
        """
        color = kwargs.get("color", "white")
        show_edges = kwargs.get("show_edges", True)
        edge_color = kwargs.get("edge_color", "black")
        opacity = kwargs.get("opacity", 1.0)
        line_width = kwargs.get("line_width", 1.0)

        # Extract only the outer cell boundaries (not internal triangulation)
        edges = self.mesh.extract_all_edges()

        # Show surface with cell boundaries only
        actor = self.plotter.add_mesh(
            self.mesh,
            color=color,
            show_edges=False,
            opacity=opacity,
        )
        self.actors.append(actor)

        # Add cell edges separately
        if show_edges:
            edge_actor = self.plotter.add_mesh(
                edges,
                color=edge_color,
                line_width=line_width,
                style="wireframe",
            )
            self.actors.append(edge_actor)

    def add_mesh_outline(self) -> None:
        """Add an outline of the mesh domain."""
        outline = self.mesh.outline()
        self.plotter.add_mesh(outline, color="black", line_width=2.0)

    def clear(self) -> None:
        """Clear all actors from the plotter."""
        self.plotter.clear()
        self.actors = []

    def show(self, **kwargs: Any) -> None:
        """
        Display the visualization.

        Args:
            **kwargs: Additional options:

                view (str): Camera view ("xy", "xz", "yz", "iso"). Default "xy".
                title (str): Title text to display.
                screenshot (str | Path): Path to save screenshot (png, jpg, eps, svg).
                    Default None.
        """
        view = kwargs.get("view", "xy")
        screenshot = kwargs.get("screenshot", None)
        title = kwargs.get("title", None)

        # Set camera view
        if view == "xy":
            self.plotter.view_xy()
            self.plotter.enable_parallel_projection()
        elif view == "xz":
            self.plotter.view_xz()
            self.plotter.enable_parallel_projection()
        elif view == "yz":
            self.plotter.view_yz()
            self.plotter.enable_parallel_projection()
        elif view == "iso":
            self.plotter.view_isometric()

        # Set the title if provided
        if title:
            self.plotter.add_title(title)

        # Save screenshot if requested
        if screenshot is not None:
            # Convert to Path if string
            screenshot_path = Path(screenshot)
            file_ext = screenshot_path.suffix.lower().lstrip(".")

            if file_ext in ["eps", "svg"]:
                self.plotter.save_graphic(str(screenshot_path), raster=False)
            else:
                self.plotter.screenshot(str(screenshot_path))

        # Show the plot
        self.plotter.show(jupyter_backend="static")
