import cv2
import importlib
import tensorflow as tf

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout,  QLabel, QPushButton, QFrame, QScrollArea, QMessageBox
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

from tools.registry import tools_dict

"""Home page style"""
class HomePage(QWidget):
    """
    Initial application page.
    """

    def __init__(self, load_tool_callback):
        super().__init__()

        self.load_tool_callback = load_tool_callback
        self.tool_widgets = {}

        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignTop)
        main_layout.setContentsMargins(60, 40, 60, 40)
        main_layout.setSpacing(25)

        # Title
        title = QLabel("FlowML-Toolkit")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""font-size: 32px;
                               font-weight: bold;""")

        main_layout.addWidget(title)

        # Description
        description = QLabel("A toolkit containing machine learning applications for problems " \
                             "related to fluid flow simulations, experiments and analysis.")
        description.setWordWrap(True)
        description.setAlignment(Qt.AlignCenter)
        description.setStyleSheet("""font-size: 16px;
                                     color: #444;""")

        main_layout.addWidget(description)

        # Available tools title
        tools_title = QLabel("Available Applications")
        tools_title.setStyleSheet("""font-size: 22px;
                                     font-weight: bold;
                                     margin-top: 20px;""")

        main_layout.addWidget(tools_title)

        # Available tools list
        for key, tool in tools_dict.items():
            container = QFrame()
            container.setStyleSheet("""QFrame { border: 1px solid #d0d0d0;
                                                border-radius: 10px;
                                                background-color: #f8f8f8; }""")

            container_layout = QVBoxLayout()
            container_layout.setContentsMargins(20, 15, 20, 15)
            container_layout.setSpacing(10)

            # Tool title button
            title_button = QPushButton(tool["name"])
            title_button.setStyleSheet("""QPushButton { text-align: left;
                                                        border: none;
                                                        font-size: 18px;
                                                        font-weight: bold;
                                                        background-color: transparent;
                                                        padding: 5px; }
                                          QPushButton:hover { color: #1565c0; }""")
            title_button.setCursor(Qt.PointingHandCursor)

            container_layout.addWidget(title_button)

            # Expandable area
            details_widget = QWidget()
            details_layout = QVBoxLayout()
            details_layout.setContentsMargins(10, 5, 10, 5)

            details_widget.setVisible(False)

            # Tool description
            tool_description = QLabel(tool["description"])
            tool_description.setWordWrap(True)
            tool_description.setStyleSheet("""font-size: 14px;
                                              color: #444;""")

            details_layout.addWidget(tool_description)

            # Open tool button
            open_button = QPushButton("Open Tool")
            open_button.setFixedWidth(140)
            open_button.setStyleSheet("""QPushButton { background-color: #1565c0;
                                                       color: white;
                                                       border-radius: 6px;
                                                       padding: 8px;
                                                       font-weight: bold; }
                                         QPushButton:hover { background-color: #0d47a1; }""")
            open_button.clicked.connect(lambda checked=False, k=key: self.load_tool_callback(k))

            details_layout.addWidget(open_button)

            details_widget.setLayout(details_layout)

            container_layout.addWidget(details_widget)

            # Toggle behavior
            title_button.clicked.connect(lambda checked=False, w=details_widget: w.setVisible(not w.isVisible()))

            container.setLayout(container_layout)

            main_layout.addWidget(container)

        # Footer
        footer = QLabel("More machine learning applications for fluid flow problems " \
                        "will be added in future versions.")
        footer.setAlignment(Qt.AlignCenter)
        footer.setWordWrap(True)
        footer.setStyleSheet("""font-size: 14px;
                                color: #666;
                                margin-top: 25px;""")

        main_layout.addWidget(footer)

        # Scroll area
        wrapper = QWidget()
        wrapper.setLayout(main_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(wrapper)
        scroll.setFrameShape(QFrame.NoFrame)

        outer_layout = QVBoxLayout()
        outer_layout.addWidget(scroll)

        self.setLayout(outer_layout)


""" Main application window """
class MainWindow(QMainWindow):

    def __init__(self):
        """
        Initializes menu items for each tool and ensures that,
        when an option is clicked, the screen displays the
        corresponding tool view.
        """
        super().__init__()
        self.setWindowTitle("FlowML-Toolkit")
        self.setWindowIcon(QIcon("logo.png"))

        self.current_widget = None

        menubar = self.menuBar()
        menubar.setStyleSheet("""QMenuBar { background-color: #f3f4f6;
                                            color: #111827;
                                            padding: 4px;
                                            border: 1px solid #bfc5cc; }

                                 QMenuBar::item { background: transparent;
                                                  padding: 6px 12px;
                                                  margin: 2px;
                                                  border-radius: 4px; }

                                 QMenuBar::item:selected { background-color: #bcc2c9; }""")

        home_action = menubar.addAction("Home")
        home_action.triggered.connect(self.go_home)
        tool_menu = menubar.addMenu("Tools")
        help_action = menubar.addAction("Help")
        help_action.triggered.connect(self.show_help)
        exit_action = menubar.addAction("Exit")
        exit_action.triggered.connect(self.close)

        # Menu items
        for key, tool in tools_dict.items():
            action = tool_menu.addAction(tool["name"])
            action.triggered.connect(lambda checked=False, k=key: self.load_tool(k))

        # Home page
        self.home_page = HomePage(self.load_tool)

        self.setCentralWidget(self.home_page)

    def load_tool(self, tool_key):
        """
        Loads one of the tools.

        :param tool_key: tool identifier
        """

        tool = tools_dict[tool_key]

        # Removes current widget
        if self.current_widget:
            self.current_widget.deleteLater()

        # Creates tool widget
        ViewClass = self.load_class(tool["view"])

        self.current_widget = ViewClass(tool["params"])

        self.setCentralWidget(self.current_widget)

    def load_class(self, path):
        """
        Loads the class responsible for handling a task view.

        :param path: path to the class
        """

        module_path, class_name = path.rsplit(".", 1)

        module = importlib.import_module(module_path)

        return getattr(module, class_name)

    def go_home(self):
        """
        Returns to the initial application page.
        """

        if self.current_widget:
            self.current_widget.deleteLater()
            self.current_widget = None

        self.home_page = HomePage(self.load_tool)

        self.setCentralWidget(self.home_page)

    def show_help(self):
        """
        Displays a help popup explaining how to access the available tools.
        """
        QMessageBox.information(self, "Help", ("FlowML-Toolkit provides machine learning tools related "
                                               "to fluid flow applications.\n\n"
                                               "There are two ways to open a tool:\n\n"
                                               "1. Through the top menu:\n"
                                               "   Tools → Select the desired application.\n\n"
                                               "2. Through the home screen:\n"
                                               "   Click the tool name to expand its description "
                                               "and then press 'Open Tool'.\n\n"
                                               "Additional applications will be added in future versions."))


app = QApplication()

window = MainWindow()

window.showMaximized()

app.exec()