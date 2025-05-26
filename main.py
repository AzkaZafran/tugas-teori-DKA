import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# semesta input
jam_belajar_per_hari = ctrl.Antecedent(np.arange(0, 8, 0.1, dtype=float), "jam_belajar_per_hari")
# jam_sosmed_per_hari = ctrl.Antecedent(np.arange(0, 5, 0.1, dtype=float), "jam_sosmed_per_hari")
jam_tidur_per_hari = ctrl.Antecedent(np.arange(0, 9, 0.1, dtype=float), "jam_tidur_per_hari")
presentase_kehadiran = ctrl.Antecedent(np.arange(0, 100, 0.1, dtype=float), "presentase_kehadiran")

# semesta output
nilai_ujian = ctrl.Consequent(np.arange(0, 100, 0.1, dtype=float), "nilai_ujian")

# membership function
jam_belajar_per_hari["kurang"] = fuzz.trapmf(jam_belajar_per_hari.universe, [0, 0, 5, 6])
jam_belajar_per_hari["cukup"] = fuzz.trapmf(jam_belajar_per_hari.universe, [5, 6, 8, 8])

# jam_sosmed_per_hari["biasa"] = fuzz.trapmf(jam_sosmed_per_hari.universe, [0, 0, 2, 3])
# jam_sosmed_per_hari["rata_rata"] = fuzz.trimf(jam_sosmed_per_hari.universe, [2, 3, 4])
# jam_sosmed_per_hari["biasa"] = fuzz.trapmf(jam_sosmed_per_hari.universe, [3, 4, 5, 5])

jam_tidur_per_hari["kurang"] = fuzz.trapmf(jam_tidur_per_hari.universe, [0, 0, 6, 7])
jam_tidur_per_hari["cukup"] = fuzz.trapmf(jam_tidur_per_hari.universe, [6, 7, 9, 9])

presentase_kehadiran["jarang"] = fuzz.trapmf(presentase_kehadiran.universe, [0, 0, 80, 90])
presentase_kehadiran["sering"] = fuzz.trapmf(presentase_kehadiran.universe, [80, 90, 100, 100])

nilai_ujian["jelek"] = fuzz.trapmf(nilai_ujian.universe, [0, 0, 70, 80])
nilai_ujian["bagus"] = fuzz.trapmf(nilai_ujian.universe, [70, 80, 85, 90])
nilai_ujian["sangat_bagus"] = fuzz.trapmf(nilai_ujian.universe, [85, 90, 100, 100])

# fuzzy rules
rule1 = ctrl.Rule(jam_belajar_per_hari["kurang"] & jam_tidur_per_hari["kurang"] & presentase_kehadiran["jarang"], nilai_ujian["jelek"])
rule2 = ctrl.Rule(jam_belajar_per_hari["kurang"] & jam_tidur_per_hari["kurang"] & presentase_kehadiran["sering"], nilai_ujian["jelek"])
rule3 = ctrl.Rule(jam_belajar_per_hari["kurang"] & jam_tidur_per_hari["cukup"] & presentase_kehadiran["jarang"], nilai_ujian["jelek"])
rule4 = ctrl.Rule(jam_belajar_per_hari["kurang"] & jam_tidur_per_hari["cukup"] & presentase_kehadiran["sering"], nilai_ujian["bagus"])
rule5 = ctrl.Rule(jam_belajar_per_hari["cukup"] & jam_tidur_per_hari["kurang"] & presentase_kehadiran["jarang"], nilai_ujian["jelek"])
rule6 = ctrl.Rule(jam_belajar_per_hari["cukup"] & jam_tidur_per_hari["kurang"] & presentase_kehadiran["sering"], nilai_ujian["bagus"])
rule7 = ctrl.Rule(jam_belajar_per_hari["cukup"] & jam_tidur_per_hari["cukup"] & presentase_kehadiran["jarang"], nilai_ujian["bagus"])
rule8 = ctrl.Rule(jam_belajar_per_hari["cukup"] & jam_tidur_per_hari["cukup"] & presentase_kehadiran["sering"], nilai_ujian["sangat_bagus"])

nilai_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8])
nilai_simulation = ctrl.ControlSystemSimulation(nilai_ctrl)

data_siswa = pd.read_csv("C:\\Users\\Azka Zafran\\Documents\\kuliah telkom semester 4\\tugas dasar kecerdasan artifisial\\vscode\\fuzzy-logic\\student_habits_performance.csv")

mae = 0
i = 0
for index, row in data_siswa.iterrows():
    nilai_simulation.input["jam_belajar_per_hari"] = row["study_hours_per_day"]
    nilai_simulation.input["jam_tidur_per_hari"] = row["sleep_hours"]
    nilai_simulation.input["presentase_kehadiran"] = row["attendance_percentage"]
    nilai_simulation.compute()
    actual = row["exam_score"]
    predicted = nilai_simulation.output["nilai_ujian"]
    mae += abs((actual - predicted))
    i += 1

mae = mae / i

print("Mean Absolute Error: " + str(mae))