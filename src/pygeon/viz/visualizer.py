import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, cast

try:
    import pyvista as pv

    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False


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

        # Configure PyVista for LaTeX rendering
        pv.global_theme.font.family = "times"
        pv.global_theme.font.label_size = 14
        pv.global_theme.font.title_size = 16

        self.plotter: "pv.Plotter" = pv.Plotter()

        # Load the PVD file
        self.meshes: dict[str, "pv.DataSet"] = {}
        self.active_mesh_name: str = ""
        self.time_values: list[float] = []
        self.current_time: float = 0.0
        self._load_pvd(file_name, time_step)

        self.actors: list[Any] = []

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
        try:
            tree = ET.parse(str(file_path))
            root = tree.getroot()
        except Exception as e:
            raise ValueError(f"Failed to parse PVD file: {e}")

        # Extract all DataSet entries from the Collection
        datasets = root.findall(".//DataSet")
        if not datasets:
            raise ValueError("No DataSet entries found in PVD file.")

        # Parse timesteps and file paths
        time_data = []
        for dataset in datasets:
            timestep = float(dataset.get("timestep", 0))
            file_ref = dataset.get("file")
            if file_ref:
                time_data.append((timestep, file_ref))

        # Sort by timestep
        time_data.sort(key=lambda x: x[0])

        if not time_data:
            raise ValueError("No valid timestep data found in PVD file.")

        # Validate time_step index
        n_steps = len(time_data)
        if time_step < 0 or time_step >= n_steps:
            raise ValueError(f"Time step {time_step} out of range [0, {n_steps - 1}]")

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
                    mesh_path = base_dir / file_ref
                    try:
                        mesh = cast("pv.DataSet", pv.read(str(mesh_path)))
                        self.meshes[f"Mesh_dim{dim}"] = mesh
                    except Exception as e:
                        raise ValueError(f"Failed to read mesh file {file_ref}: {e}")

        # Store time information
        self.time_values = [t[0] for t in time_data]
        self.current_time = time_data[time_step][0]

        # Set the first mesh as active
        if self.meshes:
            self.active_mesh_name = list(self.meshes.keys())[0]
        else:
            raise ValueError("No valid grids found in the PVD file.")

    @staticmethod
    def _extract_dimension_from_filename(filename: str) -> int | None:
        """
        Extract spatial dimension from filename.

        Assumes format like 'sol_2_000000.vtu' where _2_ indicates dimension 2.

        Args:
            filename (str): The VTU filename.

        Returns:
            int | None: The dimension (0, 1, 2, or 3) or None if not found.
        """
        # Split by underscore and look for dimension indicator
        parts = filename.replace(".vtu", "").split("_")
        for part in parts:
            if part.isdigit():
                dim = int(part)
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
        if not self.time_values:
            raise ValueError("No time steps available in this file.")

        if time_index < 0 or time_index >= len(self.time_values):
            raise ValueError(
                f"Time index {time_index} out of range [0, {len(self.time_values) - 1}]"
            )

        # Clear and reload at new time step
        self.clear()
        self.meshes = {}
        self.time_values = []
        self.current_time = 0.0
        self._load_pvd(self.file_path, time_index)

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

    def scalar_field(self, field_name: str, **kwargs: Any) -> None:
        """
        Visualize a scalar field with color mapping.

        Args:
            field_name (str): Name of the scalar field in the mesh.
            **kwargs: Additional options:
                cmap (str): Colormap name. Default "rainbow".
                show_edges (bool): Show mesh edges. Default True.
                edge_color (str): Color of edges. Default "gray".
                line_width (float): Width of edge lines. Default 1.0.
                scalar_bar_args (dict): Scalar bar configuration. Default
                    {"title": field_name}.
        """
        cmap = kwargs.get("cmap", "rainbow")
        show_edges = kwargs.get("show_edges", True)
        edge_color = kwargs.get("edge_color", "gray")
        line_width = kwargs.get("line_width", 1.0)
        scalar_bar_args = kwargs.get(
            "scalar_bar_args", {"title": field_name, "vertical": True}
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
            cast(Any, self.plotter.view_xy)()
            cast(Any, self.plotter.enable_parallel_projection)()
        elif view == "xz":
            cast(Any, self.plotter.view_xz)()
        elif view == "yz":
            cast(Any, self.plotter.view_yz)()
        elif view == "iso":
            cast(Any, self.plotter.view_isometric)()

        # Set the title if provided
        if title:
            self.plotter.add_title(title)

        # Add time information if available
        if self.time_values:
            time_text = f"t = {self.current_time:.4g}"
            self.plotter.add_text(
                time_text,
                position="lower_right",
                font_size=14,
                color="black",
            )

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
