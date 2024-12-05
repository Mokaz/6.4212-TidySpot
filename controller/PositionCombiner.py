from pydrake.all import BasicVector, Context, LeafSystem


class PositionCombiner(LeafSystem):
    """A utility class that combine the position of the base of Spot (3 dims)
    and the position of the arm of Spot (7 dims) into a single position (10 dims).
    """

    def __init__(self):
        LeafSystem.__init__(self)
        self.commanded_base_position_input = self.DeclareVectorInputPort("commanded_base_position", 3)
        self.commanded_arm_position_input = self.DeclareVectorInputPort("commanded_arm_position", 7)
        self.DeclareVectorOutputPort("spot_commanded_state", 10, self._combine_base_and_arm_states)

    def _combine_base_and_arm_states(self, context: Context, output: BasicVector):
        base_position = self.get_base_position_input_port().Eval(context)
        arm_position = self.get_arm_position_input_port().Eval(context)
        output.SetFromVector([*base_position, *arm_position])

    def connect_ports(self, builder): # Add modules as parameters to be connected to the PositionCombiner
        # builder.Connect() # TODO: Connect stuff to the PositionCombiner
        pass