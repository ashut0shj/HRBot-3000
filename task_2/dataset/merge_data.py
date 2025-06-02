
import pandas as pd


hr_data = pd.read_csv("employee_records.csv")
survey_data = pd.read_csv("employee_surveys.csv")


merged_data = pd.merge(
    hr_data,
    survey_data,
    left_on="EmpID",
    right_on="Employee ID",
    how="left"
)


merged_data["Tenure"] = (pd.to_datetime("today") - pd.to_datetime(merged_data["StartDate"])).dt.days / 365


merged_data.to_csv("merged_employee_data.csv", index=False)