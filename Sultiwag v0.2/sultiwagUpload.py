from kivymd.app import MDApp
from kivy.lang import Builder
from plyer import filechooser


kv = """
MDFloatLayout:
    MDRaisedButton:
        text: "Upload"
        pos_hint: {"center_x": .5, "center_y": .4}
        on_release:
            app.file_chooser()

    MDLabel:
        id: selected_path
        text: ""
        halign: "center"            
"""

class FileChooser(MDApp):

    def build(self):
        return self.file_chooser()

    def file_chooser(self):
        filechooser.open_file(on_selection=self.selected)
    
    def selected(self, selection):
        if selection:
            self.root.ids.selected_path.text = selection[0]

if __name__ == "__main__":
    FileChooser().run()