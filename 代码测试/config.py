import os
import sys

from PyQt5 import uic
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSpacerItem, QSizePolicy, QApplication, QLabel
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QIcon

img_dir = '代码测试/UI/imgs'

home_config = [
    {
        'type': 'block',
        'name': 'Plan',
        'buttons': [
            {'text': 'scan', 'icon': f'{img_dir}/scan.png', 'hover_icon': f'{img_dir}/scan-1.png', 'size': (56, 76)},
            {'text': 'plan', 'icon': f'{img_dir}/plan.png', 'hover_icon': f'{img_dir}/plan-1.png', 'size': (56, 76),
             'checkable': True},
            {"text": "save", "icon": f"{img_dir}/save.png", "hover_icon": f"{img_dir}/save.png", "size": (56, 76)}
            # {'text':'export', 'icon': f'{img_dir}/export.png', 'hover_icon': f'{img_dir}/export-1.png'},
            # {'text':'report', 'icon': f'{img_dir}/report.png', 'hover_icon': f'{img_dir}/report-1.png'}
        ]
    },
    {
        'type': 'block',
        'name': 'Robot Control',
        'buttons': [
            {'text': 'init_state', 'icon': f'{img_dir}/init.png', 'hover_icon': f'{img_dir}/init-1.png',
             'size': (56, 76)},
            {'text': 'auto', 'icon': f'{img_dir}/auto.png', 'hover_icon': f'{img_dir}/auto-1.png', 'size': (56, 76)},
            {'text': 'grav', 'icon': f'{img_dir}/drag.png', 'hover_icon': f'{img_dir}/drag-1.png', 'size': (56, 76)},
            {'text': 'stop', 'icon': f'{img_dir}/stop.png', 'hover_icon': f'{img_dir}/stop-1.png', 'size': (56, 76)},
        ]
    }]


class TextIconButton(QWidget):
    toggled = pyqtSignal(object)

    def __init__(
            self, text, icon, hover_icon=None,
            size=(64, 84), action=None,
            checkable=False, checked=False,
            show_bg=True, disable_flag=False,
            hover_enable=True, parent=None
    ):
        super(TextIconButton, self).__init__(parent)
        self.text = text
        self.icon = icon
        self._size = size
        self.hover_icon = hover_icon or icon
        self.action = action
        self.checkable = checkable
        self.checked = checked
        self.show_bg = show_bg
        self.disabled = False
        self.hover_enable = hover_enable
        self.disable_flag = disable_flag

        self.initUI()

    def initUI(self):
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_StyledBackground)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        w, h = self._size
        self._layout.setSpacing(4)
        self.text_size = 14 if w == 64 else 14

        self._icon_label = QLabel(self)
        self._icon_label.setFixedSize(56, 56)
        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignCenter)
        hlayout = QHBoxLayout()
        hlayout.setContentsMargins((w - 56) // 2, 0, (w - 56) // 2, 0)
        hlayout.addWidget(self._icon_label)
        self._layout.addLayout(hlayout)
        self._layout.addWidget(self._label)

        if self.text:
            self._label.show()
            self._label.setText(self.text)
            self.setFixedSize(w, h)
        else:
            self._label.hide()
            self.setFixedSize(w, w)

        self.update_style()

    def set_disabled(self, disabled):
        if disabled:
            self.update_style("normal")
        self.disabled = disabled

    def update_style(self, state="normal"):
        if self.disabled:
            return

        if self.checkable and state != "hover":
            state = "hover" if self.checked else "normal"

        if state == "normal":
            icon = self.icon
            bg_color = "rgba(255,255,255,0.15)"
            radius = ""
            text_color = "rgba(255,255,255, 0.85)"
        else:
            icon = self.hover_icon
            bg_color = "rgba(255,255,255,0.35)"
            radius = "border-top-right-radius: 8px;"
            text_color = "rgb(0, 156, 244)"

        bg_color = bg_color if self.show_bg else "transparent"
        self._icon_label.setStyleSheet(
            f"background-color:{bg_color};border-image: url({icon});border: none; {radius}"
        )
        self._label.setStyleSheet(
            f"font-family:'Roboto';font-weight:400;background:transparent; border:none; color: {text_color}; font-size:{self.text_size}px;")

    def mousePressEvent(self, event):
        pass

    def set_checked(self, checked):
        self.checked = checked
        self.update_style()

    def enterEvent(self, e):
        pass

    def leaveEvent(self, e):
        if not self.hover_enable:
            return
        self.update_style()


class NavBlockWidget(QWidget):
    def __init__(self, block_name, buttons, show_block_name=True, column=4, parent=None):
        super(NavBlockWidget, self).__init__(parent)
        self.only_one_checked = False
        self.show_name = show_block_name
        self.column_count = column
        self.initUI(block_name, buttons)

    def initUI(self, block_name, buttons):
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_StyledBackground)

        uic.loadUi('代码测试/UI/nav_block_widget.ui', self)

        btn_count = len(buttons)
        cc = self.column_count
        if btn_count <= 0:
            return

        self.name_label.setStyleSheet('background: transparent;font-size: 24px;font-family: Roboto;color: white;')
        self.name_label.setText(block_name)

        name_height = 52
        if not self.show_name:
            self.name_label.hide()
            name_height = 16

        self.btns = []
        self.only_one_checked = buttons[0].get('button_group', False)
        # 按钮高度，有文字信息的为84，没有文字信息的为64
        btn_height = 64 if buttons[0]['text'] == '' else 84

        # 根据行数计算btn widget的高度
        rows = btn_count // cc + (0 if btn_count % cc == 0 else 1)
        spacing = 16
        btn_widget_height = rows * btn_height + (rows - 1) * spacing
        self.btn_widget.setFixedHeight(btn_widget_height)

        self.setFixedHeight(name_height + btn_widget_height)
        # 遍历buttons，添加按钮
        for i, btn in enumerate(buttons):
            disabeld_flag = btn['text'] == "auto"
            ti_btn = TextIconButton(
                btn['text'], btn['icon'], btn['hover_icon'],
                checkable=btn.get('checkable', False),
                checked=btn.get('checked', False),
                size=btn.get('size', (64, btn_height)),
                disable_flag=disabeld_flag,
                parent=self
            )
            self.btns.append(ti_btn)
            ti_btn.toggled.connect(self.update_btn_state)
            self.btn_layout.addWidget(ti_btn, i//cc, i%cc)

        # 添加弹簧
        self.btn_layout.addItem(
            QSpacerItem(0, 0, hPolicy=QSizePolicy.Policy.Expanding),
            0, 4)

    def update_btn_state(self):
        btn = self.sender()
        [_btn.set_checked(False) for _btn in self.btns]
        btn.set_checked(True)


def create_tool_widget(data, parent=None):
    if data['type'] == 'block':
        return NavBlockWidget(
            data['name'], data.get('buttons', []),
            data.get('show_block_name', True),
            parent=parent
        )
    else:
        return None


class NavPageWidget(QWidget):
    ''' home and robot page '''

    def __init__(self, config: list, parent=None):
        super(NavPageWidget, self).__init__(parent)
        self._config = config
        self.initUI()

    def initUI(self):
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_StyledBackground)
        self.setFixedWidth(328)

        # 创建竖直布局，设置间距20，边距为0
        self._layout = QVBoxLayout(self)
        self._layout.setSpacing(20)
        self._layout.setContentsMargins(0, 16, 0, 0)

        # self._line_tip = LineTipWidget(self)
        # self._line_tip.tip_closed.connect(self.hide_line_tip)
        # self._line_tip.hide()
        # self._layout.addWidget(self._line_tip)

        # 遍历 config 字典数据，创建block
        for item in self._config:
            self._layout.addWidget(
                create_tool_widget(item, self)
            )
        self._layout.addSpacerItem(
            QSpacerItem(0, 0, vPolicy=QSizePolicy.Policy.Expanding))

    # def show_line_tip(self, message, state):
    #     if message != self._line_tip.message:
    #         self._line_tip.show_tip(message, state)
    #         self._layout.setContentsMargins(0,16,0,0)

    # def hide_line_tip(self):
    #     self._line_tip.hide()
    #     self._layout.setContentsMargins(0,0,0,0)

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    win = NavPageWidget(home_config)
    win.show()
    app.exec()
