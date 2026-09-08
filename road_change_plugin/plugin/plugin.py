from .signals import TaskSignals, forward
from .controller import Controller
from .widget import RoadChangeWidget


class RoadChangePlugin(TaskSignals):
    plugin_id = "road_change"
    name = "道路变化检测"
    version = "1.0.0"
    api_version = "1"

    def __init__(self):
        super().__init__()
        self._controller = Controller(self)
        forward(self._controller, self)

    def create_widget(self, parent=None):
        return RoadChangeWidget(self._controller, parent)

    def shutdown(self):
        self._controller.shutdown()


def create_plugin():
    return RoadChangePlugin()
