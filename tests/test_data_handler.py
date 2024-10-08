import unittest
import pandas as pd
import numpy as np
from vasp.data_handler import DATA_HANDLER
import os
import laspy
from plyfile import PlyData

class Test_DATA_HANDLER_WithRealData(unittest.TestCase):
    def setUp(self):
        # Pfad zur Testdatei
        
        self.test_file_path = r"tests\test_data\vasp_in.laz"
        self.test_file_paths = [
            r"tests\test_data\vasp_in.laz",
            r"tests\test_data\vasp_in.laz"
        ]

        self.test_output_file = r"tests\test_data\output_test.laz"
        self.test_output_ply = r"tests\test_data\output_test.ply"

        # Überprüfen, ob die Testdatei existiert
        if not os.path.exists(self.test_file_path):
            self.fail(f"Testdatei {self.test_file_path} existiert nicht.")
        
        for file in self.test_file_paths:
            if not os.path.exists(file):
                self.fail(f"Testdatei {file} existiert nicht.")

        # Instanz der DATA_HANDLER-Klasse erstellen
        self.handler = DATA_HANDLER(infiles=[self.test_file_path])
        self.handler.load_las_files()

    def test_load_las_files_empty_list(self):
        handler = DATA_HANDLER(infiles=[])
        with self.assertRaises(ValueError):
            handler.load_las_files()

    def test_load_las_files_invalid_path(self):
        handler = DATA_HANDLER(infiles=["non_existent_file.laz"])
        with self.assertRaises(FileNotFoundError):
            handler.load_las_files()

    def test_load_las_files(self):
        # Methode aufrufen
        self.handler.load_las_files()
        
        # Überprüfen, ob der DataFrame erstellt wurde
        self.assertIsNotNone(self.handler.df, "Der DataFrame wurde nicht erstellt.")
        
        # Überprüfen, ob der DataFrame nicht leer ist
        self.assertFalse(self.handler.df.empty, "Der DataFrame ist leer.")
        
        # Überprüfen, ob die erwarteten Spalten vorhanden sind
        expected_columns = ['X', 'Y', 'Z'] + self.handler.attributes
        self.assertListEqual(list(self.handler.df.columns), expected_columns, "Die Spalten des DataFrames stimmen nicht überein.")
        
        # Optional: Überprüfen Sie einige Werte im DataFrame
        # Hier können Sie spezifische Überprüfungen basierend auf Ihrem Testdatensatz hinzufügen

    def test_load_las_files_multiple_files(self):
        # Falls Sie mehrere Testdateien haben, können Sie diesen Test erweitern
        
        for path in self.test_file_paths:
            if not os.path.exists(path):
                self.fail(f"Testdatei {path} existiert nicht.")
        
        # Instanz mit mehreren Dateien erstellen
        handler = DATA_HANDLER(infiles=self.test_file_paths)
        handler.load_las_files()
        
        # Überprüfungen wie zuvor
        self.assertIsNotNone(handler.df, "Der DataFrame wurde nicht erstellt.")
        self.assertFalse(handler.df.empty, "Der DataFrame ist leer.")

    def test_save_as_las(self):
        # Pfad für die Ausgabedatei
        
        handler = DATA_HANDLER(infiles=self.test_file_path)
        handler.load_las_files()
        # Speichere die Daten in eine neue LAS-Datei
        handler.save_as_las(self.test_output_file)
        
        # Lese die gespeicherte Datei
        with laspy.open(self.test_output_file) as las_file:
            las = las_file.read()
        
        # Vergleiche die Koordinaten
        np.testing.assert_array_almost_equal(las.x, handler.df["X"], decimal=5)
        np.testing.assert_array_almost_equal(las.y, handler.df["Y"], decimal=5)
        np.testing.assert_array_almost_equal(las.z, handler.df["Z"], decimal=5)
        
        # Vergleiche zusätzliche Attribute
        for attr in handler.attributes:
            if hasattr(las, attr):
                np.testing.assert_array_equal(getattr(las, attr), handler.df[attr])
            elif attr in las.point_format.extra_dimension_names:
                # Falls das Attribut als extra Dimension gespeichert wurde
                las_attr = las[attr]
                np.testing.assert_array_equal(las_attr, handler.df[attr])
            else:
                self.fail(f"Attribut {attr} wurde nicht in der gespeicherten LAS-Datei gefunden.")
        
        # Aufräumen: Entferne die erzeugte Ausgabedatei
        del las
        os.remove(self.test_output_file)

    def test_save_as_ply(self):
        def _remove_ply(test_output_ply):
            os.remove(test_output_ply)
        # Define parameters for save_as_ply
        voxel_size = 1.0  # Example value
        shift_to_center = False

        # Save the data to a .ply file
        self.handler.save_as_ply(self.test_output_ply, voxel_size, shift_to_center)

        # Read the saved .ply file using a context manager
        with open(self.test_output_ply, 'rb') as file:
            ply_data = PlyData.read(file, mmap=False)
            # Extract the saved data
            vertex_data = ply_data['vertex'].data

        ply_df = pd.DataFrame(vertex_data)

        # Verify if the number of points matches
        self.assertEqual(len(ply_df), len(self.handler.df) * 8, "Number of planes and points do not match.")

        # If color values are present, verify them
        color_attributes = ['red', 'green', 'blue']
        for color in color_attributes:
            if color in self.handler.df.columns:
                self.assertIn(color, ply_df.columns, f"{color.capitalize()} value is missing in the .ply file.")
                np.testing.assert_array_equal(ply_df[color][::8], self.handler.df[color])
            else:
                self.assertNotIn(color, ply_df.columns, f"{color.capitalize()} value should not be present in the .ply file.")
        
        _remove_ply(self.test_output_ply)



if __name__ == '__main__':
    unittest.main()