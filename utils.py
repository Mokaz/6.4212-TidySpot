from pydrake.all import Diagram
from typing import BinaryIO, Optional, Union, Tuple
import pydot
import logging

### Filter out Drake warnings ###
class DrakeWarningFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "has its own materials, but material properties have been defined as well" in msg or \
           "material [ 'wire_088144225' ] not found in .mtl" in msg:
            return False
        return True

    
def export_diagram_as_svg(diagram: Diagram, file: Union[BinaryIO, str]) -> None:
    if type(file) is str:
        file = open(file, "bw")
    graphviz_str = diagram.GetGraphvizString()
    svg_data = pydot.graph_from_dot_data(
        diagram.GetGraphvizString())[0].create_svg()
    file.write(svg_data)