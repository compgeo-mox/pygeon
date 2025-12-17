import shutil
from pathlib import Path
from typing import Any, cast

import pyvista as pv


class Visualizer:
    """
    A flexible visualization class for VTU mesh file using PyVista.

    Supports both Jupyter notebook and standalone usage with various
    visualization modes for scalar and vector fields.

    NOTE: the visualization of a vector field should be called before the one of a
    scalar field. Contour plot should be called last.
    """

    def __init__(
        self,
        file_name: str | Path,
        folder_name: str | Path = "",
    ) -> None:
        """
        Initialize the Visualizer.

        Args:
            file_name (str | Path): Name or path to the PVD file
                (extension .pvd can be omitted).
            folder_name (str | Path): Optional folder path. If provided,
                will be combined with file_name.
        """
        # Convert to Path objects if strings
        file_name = Path(file_name)

        # Construct full path as union of folder_name and file_name
        if folder_name != "":
            folder_name = Path(folder_name)
            file_name = folder_name / file_name

        # Cast to DataSet to satisfy type checkers
        self.mesh: pv.DataSet = cast(pv.DataSet, pv.read(str(file_name)))
        self.dim = self.mesh.GetMaxSpatialDimension()

        # Configure PyVista theme
        pv.global_theme.font.label_size = 14
        pv.global_theme.font.title_size = 16

        # Enable LaTeX rendering if available
        if shutil.which("latex") is not None:
            pv.global_theme.font.family = "times"
            # LaTeX is available, PyVista can use it for text rendering
        else:
            pv.global_theme.font.family = "arial"

        self.plotter = pv.Plotter()

    def _default_bar_args(self, field_name: str) -> dict[str, Any]:
        """
        Return centered right-side scalar bar args based on mesh dimension.

        Args:
            field_name (str): Name of the field in the mesh.
        """
        # Make bar size proportional to dimension and center vertically
        # Height grows with dimension but is clamped for readability
        height = 0.35 + 0.1 * self.dim  # 1D->0.45, 2D->0.55, 3D->0.65
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

    def plot_vector_field(self, field_name: str, scaling_factor: float = 1.0) -> None:
        """
        Visualize a vector field using arrows.

        Args:
            field_name (str): Name of the vector field in the mesh.
            scaling_factor (float): Scaling factor for arrows. Default 1.0.
        """
        arrows = self.mesh.glyph(
            orient=field_name, scale=field_name, factor=scaling_factor, absolute=False
        )
        self.plotter.add_mesh(arrows, color="gray", scalars=None, cmap=None)

    def plot_scalar_field(
        self, field_name: str, field_label: str = "", **kwargs: Any
    ) -> None:
        """
        Visualize a scalar field with color mapping.

        Args:
            field_name (str): Name of the scalar field in the mesh.
            field_label (str): Label for the scalar bar. Default is field_name.
            **kwargs: Additional options:

                - cmap (str): Colormap name. Default "rainbow".
                - show_edges (bool): Show mesh edges. Default True.
                - edge_color (str): Color of edges. Default "gray".
                - line_width (float): Width of edge lines. Default 1.0.
                - scalar_bar_args (dict): Scalar bar configuration.
        """
        if field_label == "":
            field_label = field_name

        cmap = kwargs.get("cmap", "rainbow")
        show_edges = kwargs.get("show_edges", True)
        edge_color = kwargs.get("edge_color", "gray")
        line_width = kwargs.get("line_width", 1.0)
        scalar_bar_args = kwargs.get(
            "scalar_bar_args", self._default_bar_args(field_label)
        )

        # Show scalar field without edges to avoid triangulation visibility
        self.plotter.add_mesh(
            self.mesh,
            scalars=field_name,
            cmap=cmap,
            show_edges=False,
            scalar_bar_args=scalar_bar_args,
        )

        # Add cell edges separately if requested
        if show_edges:
            edges = self.mesh.extract_all_edges()
            self.plotter.add_mesh(
                edges,
                color=edge_color,
                line_width=line_width,
                style="wireframe",
            )

    def plot_contour(self, field_name: str, isosurfaces: int = 10) -> None:
        """
        Create contour surfaces of a scalar field.

        NOTE: if the scalar field is a cell data, it will be converted to point data
        before doing the contour.

        Args:
            field_name (str): Name of the scalar field.
            isosurfaces (int): Number of isosurfaces. Default 10.
        """
        # Check if field is cell data, convert to point data if needed
        mesh = self.mesh
        if field_name in mesh.cell_data:
            mesh = mesh.cell_data_to_point_data()

        contours = mesh.contour(isosurfaces=isosurfaces, scalars=field_name)
        self.plotter.add_mesh(contours, color="black", line_width=2.0)

    def plot_mesh(self, **kwargs: Any) -> None:
        """
        Visualize the mesh grid without any field data.

        Args:
            **kwargs: Additional options:

                - color (str): Mesh color. Default "white".
                - show_edges (bool): Show mesh edges. Default True.
                - edge_color (str): Color of edges. Default "black".
                - opacity (float): Mesh opacity. Default 1.0.
                - line_width (float): Width of edge lines. Default 1.0.
        """
        color = kwargs.get("color", "white")
        show_edges = kwargs.get("show_edges", True)
        edge_color = kwargs.get("edge_color", "black")
        opacity = kwargs.get("opacity", 1.0)
        line_width = kwargs.get("line_width", 1.0)

        # Extract only the outer cell boundaries (not internal triangulation)
        edges = self.mesh.extract_all_edges()

        # Show surface with cell boundaries only
        self.plotter.add_mesh(
            self.mesh,
            color=color,
            show_edges=False,
            opacity=opacity,
        )

        # Add cell edges separately
        if show_edges:
            self.plotter.add_mesh(
                edges,
                color=edge_color,
                line_width=line_width,
                style="wireframe",
            )

    def show(self, **kwargs: Any) -> None:
        """
        Display the visualization.

        Args:
            **kwargs: Additional options:

                - view (str): Camera view ("xy", "xz", "yz", "iso"). Default "xy".
                - title (str): Title text to display.
                - screenshot (str | Path): Path to save screenshot (png, jpg, eps, svg).
        """
        view = kwargs.get("view", "xy")
        screenshot = kwargs.get("screenshot", None)
        title = kwargs.get("title", None)

        # Set camera view
        renderer = self.plotter.renderer
        if view == "xy":
            renderer.view_xy()
            renderer.enable_parallel_projection()
        elif view == "xz":
            renderer.view_xz()
            renderer.enable_parallel_projection()
        elif view == "yz":
            renderer.view_yz()
            renderer.enable_parallel_projection()
        elif view == "iso":
            renderer.view_isometric()

        # Set the title if provided
        if title:
            self.plotter.add_title(title)

        # Render and save/show
        if screenshot is not None:
            # Ensure off-screen rendering so screenshot works without a prior .show()
            off_screen_prev = getattr(self.plotter, "off_screen", False)
            global_off_prev = getattr(pv, "OFF_SCREEN", False)
            pv.OFF_SCREEN = True
            self.plotter.off_screen = True

            # Ensure the underlying VTK render window is off-screen to prevent flicker
            ren_win = getattr(self.plotter, "ren_win", None)
            if ren_win is not None and hasattr(ren_win, "SetOffScreenRendering"):
                ren_win.SetOffScreenRendering(True)

            # Convert to Path if string
            screenshot_path = Path(screenshot)
            file_ext = screenshot_path.suffix.lower().lstrip(".")

            # Render first
            self.plotter.render()
            # Save based on format
            if file_ext in ["eps", "svg"]:
                self.plotter.save_graphic(str(screenshot_path), raster=False)
            else:
                self.plotter.screenshot(str(screenshot_path))

            self.plotter.close()
            self.plotter.off_screen = off_screen_prev
            pv.OFF_SCREEN = global_off_prev
            # Recreate plotter for future use (e.g., if show() is called again)
            self.plotter = pv.Plotter()
        else:
            # Show the plot interactively
            if self.plotter.notebook:
                self.plotter.show(jupyter_backend="static")
            else:
                self.plotter.show()
