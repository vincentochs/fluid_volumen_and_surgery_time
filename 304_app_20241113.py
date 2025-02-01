# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 22:10:54 2024

@author: Vincent Ochs
"""

###############################################################################
# Import libraries

# App
import streamlit as st
from streamlit_option_menu import option_menu

# Utils
import pandas as pd
import pickle as pkl
import numpy as np
from itertools import product
import joblib
import pandas as pd, numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import make_interp_spline, BSpline
# Models
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

print('Libreries loaded')

###############################################################################
# Define model architecture
class ThresholdLayer(nn.Module):
    def __init__(self):
        super(ThresholdLayer, self).__init__()
        self.threshold1_low = nn.Parameter(torch.tensor(0.0))
        self.threshold1_high = nn.Parameter(torch.tensor(1.0))
        self.threshold2_low = nn.Parameter(torch.tensor(0.0))
        self.threshold2_high = nn.Parameter(torch.tensor(1.0))

    def forward(self, feature1, feature2):
        within_threshold1 = (feature1 >= self.threshold1_low) & (feature1 <= self.threshold1_high)
        within_threshold2 = (feature2 >= self.threshold2_low) & (feature2 <= self.threshold2_high)
        risk_score = 1 - (within_threshold1 & within_threshold2).float()
        return risk_score

class RiskClassificationModel(nn.Module):
    def __init__(self, other_features_dim, hidden_dim):
        super(RiskClassificationModel, self).__init__()
        self.threshold_layer = ThresholdLayer()
        self.fc1 = nn.Linear(other_features_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim + hidden_dim, 1)

    def forward(self, feature1, feature2, other_features):
        risk_score = self.threshold_layer(feature1, feature2).view(-1, 1)
        x = F.relu(self.fc1(other_features))
        risk_score_expanded = risk_score.expand(-1, x.size(1))
        combined_features = torch.cat([x, risk_score_expanded], dim=1)
        output = torch.sigmoid(self.fc2(combined_features))
        return output
# Define function to save pytorch model for early stopping
def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)
# Define function to load best early stopping pytorch model to continue with the evaluation
def resume(model, filename):
    model.load_state_dict(torch.load(filename))
###############################################################################
# PARAMETERS SECTION
# Define operation time and fluid sume range to simulate
MINIMUM_OPERATION_TIME = 100
MINIMUM_FLUID_SUM = 100
MAXIMUM_OPERATION_TIME = 600
MAXIMUM_FLUID_SUM = 1000


# Define dictionary for model inputs names
INPUT_FEATURES = {'Sex (1: Male, 2: Female)' : {'Male' : 1,
                                                'Female' : 2},
                  'Active Smoking (1: Yes, 0: No)' : {'Yes' : 1,
                                                      'No' : 0},
                  'Alcohol abuse (1: <2 beverages/day, 2: >= 2 beverages/day, 3: No alcohol abuse, 4:Unknown)' : {'<2 beverages/day' : 1,
                                                                                                                  '>= 2 beverages/day' : 2,
                                                                                                                  'No alcohol abuse' : 3,
                                                                                                                  'Unknown' : 4},
                  'Real function CKD stages G1 (normal) to G5 (1: G1, 2: G2, 3:G3a, 4: G3b, 5: G4, 6: G5)' : {'G1' : 1,
                                                                                                              'G2' : 2,
                                                                                                              'G3a' : 3,
                                                                                                              'G3b' : 4,
                                                                                                              'G4' : 5,
                                                                                                              'G5' : 6},
                  'Liver metastasis at time of anastomosis (any) (1: Yes, 2: No, 3: Unknown)' : {'Yes' : 1,
                                                                                                 'No' : 2,
                                                                                                 'Unknown' : 3},
                  'Neoadjuvant Therapy (1=yes, 0 = no)' : {'Yes' : 1,
                                                           'No' : 0},
                  'Preoperative use of immunosuppressive drugs 2 weeks before surgery (1: Yes, 0: No, 2: Unknown)' : {'Yes' : 1,
                                                                                                                      'No' : 0,
                                                                                                                      'Unknown' : 2},
                  'Preoperative steroid use (1: Yes, 0: No, 2: Unknown)' : {'Yes' : 1,
                                                                            'No' : 0,
                                                                            'Unknown' : 2},
                  'Preoperative NSAIDs use (1: Yes, 0: No, 2: Unknown)' : {'Yes' : 1,
                                                                           'No' : 0,
                                                                           'Unknown' : 2},
                  'Preoperative blood transfusion (1: Yes, 0: No, 2: Unknown)' : {'Yes' : 1,
                                                                                  'No' : 0,
                                                                                  'Unknown' : 2},
                  'TNF Alpha Inhib (1=yes, 0=no)' : {'Yes' : 1,
                                                     'No' : 0},
                  'Charlson comorbidity index' : {str(i) : i for i in range(17)},
                  'American Society of Anesthesiologists (ASA) Score (1: ASA 1: healthy person, 2: ASA 2: mild systemic disease, 3: ASA 3: severe systemic disease, 4: ASA 4: severe systemic disease that is a constant threat to life, 5: ASA 5: a moribund person who is not ex, 6: Unknown' :  {'1: Healthy Person' : 1,
                           '2: Mild Systemic disease' : 2,
                           '3: Severe syatemic disease' : 3,
                           '4: Severe systemic disease that is a constan threat to life' : 4,
                           '5: Moribund person' : 5,
                           '6: Unkonw' : 6},
                  'Prior abdominal surgery (1: Yes, 2: No, 3: Unknown)' : {'Yes' : 1,
                                                                           'No' : 2,
                                                                           'Unknown' : 3},
                  'Indication (1: Recurrent Diverticulitis, 2: Acute Diverticulitis, 3: Ileus/Stenosis, 4: Ischemia, 5: Tumor, 6: Volvulus, 7: Morbus crohn, 8: Colitis ulcerosa, 9: Perforation (müsste perforation = yes und emergency = yes -> muss in 10 other), 10: Other, 11: Ileostoma reversal = zu 12, 12: Colostoma reversal' : {'Recurrent Diverticulitis' : 1,
                                                                                                                                                                                                                                                                                                                                           'Acute Diverticulitis' : 2,
                                                                                                                                                                                                                                                                                                                                           'Ileus/Stenosis' : 3,
                                                                                                                                                                                                                                                                                                                                           'Ischemia' : 4,
                                                                                                                                                                                                                                                                                                                                           'Tumor' : 5,
                                                                                                                                                                                                                                                                                                                                           'Volvulus' : 6,
                                                                                                                                                                                                                                                                                                                                           'Morbus crohn' : 7,
                                                                                                                                                                                                                                                                                                                                           'Colitis ulcerosa' : 8,
                                                                                                                                                                                                                                                                                                                                           'Perforation (müsste perforation = yes und emergency = yes' : 9,
                                                                                                                                                                                                                                                                                                                                           'Other' : 10,
                                                                                                                                                                                                                                                                                                                                           'Ileostoma reversal' : 11,
                                                                                                                                                                                                                                                                                                                                           'Colostoma reversal' : 12},
                  'Operation' : {'Rectosigmoid resection/sigmoidectomy' : 1,
                                 'Left hemicolectomy' : 2,
                                 'Extended left hemicolectomy' : 3, 
                                 'Right hemicolectomy' : 4, 
                                 'Extended right hemicolectomy' : 5, 
                                 'Transverse colectomy' : 6, 
                                 'Hartmann conversion' : 7, 
                                 'Ileocaecal resection' : 8, 
                                 'Total colectomy' : 9, 
                                 'High anterior resection (anastomosis higher than 12cm)' : 10, 
                                 'Low anterior resection (anastomosis 12 cm from anal average and below)' : 11, 
                                 'Abdominoperineal resection' : 12, 
                                 'Adhesiolysis with small bowel resection' : 13, 
                                 'Adhesiolysis only' : 14, 
                                 'Hartmann resection / Colostomy' : 15, 
                                 'Colon segment resection' : 16, 
                                 'Small bowl resection' : 17},
                  'Emergency surgery (1: Yes, 0: No, 2: Unknown)' : {'Yes' : 1,
                                                                     'No' : 0,
                                                                     'Unknown' : 2},
                  'Perforation (1: Yes, 0: No)' : {'Yes' : 1,
                                                   'No' : 0},
                  'Approach (1: Laparoscopic, 2: Robotic, 3: Open, 4: Conversion to open, 5: Conversion to laparoscopy, 6: Transanal (ta TME, TATA, TAMIS))' : {'1: Laparoscopic' : 1 ,
                                        '2: Robotic' : 2 ,
                                        '3: Open to open' : 3,
                                        '4: Conversion to open' : 4,
                                        '5: Conversion to laparoscopy' : 5},
                  'Type of anastomosis (1: Colon anastomosis, 2: Colorectal anastomosis, 3: Ileocolonic anastomosis, 4: Ileorectal anastomosis, 5: Ileopouch-anal, 6: Colopouch, 7: Small intestinal anastomosis, 8: Unknown)' : {'Colon anastomosis' : 1,
                                    'Colorectal anastomosis' : 2, 
                                    'Ileocolonic anastomosis' : 3, 
                                    'Ileorectal anastomosis' : 4, 
                                    'Ileopouch-anal' : 5, 
                                    'Colopouch' : 6, 
                                    'Small intestinal anastomosis' : 7, 
                                    'Unknown' : 8},
                  'Anastomotic technique (1: Stapler, 2: Hand-sewn, 3: Stapler and Hand-sewn, 4: Unknown) (alle 3 werden zu 1)' : {'1: Stapler' : 1,
                                                                                                                                   '2: Hand-sewn' : 2,
                                                                                                                                   '3: Stapler and Hand-sewn' : 3,
                                                                                                                                   '4: Unknown' : 4},
                  'Anastomotic configuration (1: end-to-end, 2: side-to-end, 3: side-to-side, 4: end-to-side, 5: Unknown)' : {'End to End' : 1,
                                                                                                                              'Side to End' : 2,
                                                                                                                              'Side to Side' : 3,
                                                                                                                              'End to Side' : 4},
                  'Protective stomy (1: Ileostomy, 2: Colostomy, 3: No protective stomy, 4: Unknown)' : {'Ileostomy' : 1,
                                                                                                         'Colostomy' : 2,
                                                                                                         'No protective stomy' : 3,
                                                                                                         'Unknown' : 4},
                  "Surgeon's experience (1: Consultant (the counsalting performed the operation, the other persons only assisted), 2: Teaching operation (Consultant with senior resident, the Resident was allowed to do part or more of the case), 3: Unknown)" : {'Consultant' : 1,
                                                                                                                                                                                                                                                                     'Teaching Operation' : 2,
                                                                                                                                                                                                                                                                     'Unknown' : 3},
                  'Total points Nutritional status' :  {str(i) : i for i in range(7)}}

###############################################################################
# Section when the app initialize and load the required information
#@st.cache_data() # We use this decorator so this initialization doesn't run every time the user change into the page
def initialize_app():   
    # Load model
    path_model = r'models'
    preprocesor_filename = r'/304_preprocesor.joblib'
    model_filename = r'/304_model_yes_weight_risk.pth'
    other_features_dim = 32
    hidden_dim = 512
    preprocesor = joblib.load(path_model + preprocesor_filename)
    model = RiskClassificationModel(other_features_dim=other_features_dim, hidden_dim=hidden_dim)
    resume(model, path_model + model_filename)
    print('File loaded -->' , path_model + model_filename)
    print('File loaded -->' , path_model + preprocesor_filename)
    
    print('App Initialized correctly!')
    
    return model , preprocesor

# Function to parser input
def parser_input(df_input , model ,preprocesor ,  initial_operation_time , initial_fluid_sum):
    
    # Encode categorical features
    for i in df_input.columns.tolist():
        if i in list(INPUT_FEATURES.keys()):
            df_input[i] = df_input[i].map(INPUT_FEATURES[i])
    print('Features encoded')
    # Put clinic number ad hoc
    df_input['data_group_encoded'] = 8
    # Define range of operation time in minutes and Fluid sum in mL to simulate
    range_operation_time = list(range(MINIMUM_OPERATION_TIME , MAXIMUM_OPERATION_TIME + 5, 5))
    range_fluid_sum = list(range(MINIMUM_FLUID_SUM , MAXIMUM_FLUID_SUM + 10 , 10))
    
    # Generate 2 separated dataframes for independent fluid / operation simulation
    df_operation = df_input.copy()
    df_fluid = df_input.copy()
    # Add initial values of operation and fluid
    df_operation['Fluid_sum'] = initial_fluid_sum
    df_fluid['Operation time (min)'] = initial_operation_time
    # Generate n times the df for simulation
    df_operation = pd.concat([df_operation] * len(range_operation_time) , ignore_index = True)
    df_fluid = pd.concat([df_fluid] * len(range_fluid_sum) , ignore_index = True)
    # Put simulated values
    df_operation['Operation time (min)'] = range_operation_time
    df_fluid['Fluid_sum'] = range_fluid_sum
    # Make predictions
    model.eval()
    with torch.no_grad():
        test_f1 = torch.tensor(df_operation['Fluid_sum'].values).float().view(-1, 1)
        test_f2 = torch.tensor(df_operation['Operation time (min)'].values).float().view(-1, 1)
        test_other = torch.tensor(df_operation.drop(columns = ['Fluid_sum' , 'Operation time (min)']).values).float()

        test_output = model(test_f1, test_f2, test_other).squeeze()
    
    df_operation['pred_proba'] = test_output.detach().numpy() * 100
    
    model.eval()
    with torch.no_grad():
        test_f1 = torch.tensor(df_fluid['Fluid_sum'].values).float().view(-1, 1)
        test_f2 = torch.tensor(df_fluid['Operation time (min)'].values).float().view(-1, 1)
        test_other = torch.tensor(df_fluid.drop(columns = ['Fluid_sum' , 'Operation time (min)']).values).float()

        test_output = model(test_f1, test_f2, test_other).squeeze()
    df_fluid['pred_proba'] = test_output.detach().numpy() * 100
    # Sort by probability
    df_operation = df_operation.sort_values(by = 'pred_proba')
    df_fluid = df_fluid.sort_values(by = 'pred_proba')
    #st.dataframe(df_operation.head(50))
    #st.write('-' * 50)
    #st.dataframe(df_fluid.head(100))
    # Make line plots for each dataframe
    st.write('Simulation for Surgery Duration:')
    x = df_operation.sort_values(by = 'Operation time (min)')['Operation time (min)'].values
    x = np.array(range_operation_time)
    x_plot = np.linspace(x.min(), x.max(), 5000) 
    y = df_operation['pred_proba'].values
    print(x)
    y_plot = make_interp_spline(x, y, k=3)
    y_plot_smooth = y_plot(x_plot)
    
    # Create figure and plot
    fig, ax = plt.subplots()
    ax.plot(x_plot, y_plot_smooth)
    ax.set_xlabel("Operation time (min)")
    ax.set_ylabel("Predicted Probability")
    st.pyplot(fig)
    
    st.write('Simulation for Fluid Volumen:')
    x = df_fluid.sort_values(by = 'Fluid_sum')['Fluid_sum'].values
    x = np.array(range_fluid_sum )
    x_plot = np.linspace(x.min(), x.max(), 5000) 
    y = df_fluid['pred_proba'].values
    print(x)
    y_plot = make_interp_spline(x, y, k=3)
    y_plot_smooth = y_plot(x_plot)
    
    # Create figure and plot
    fig, ax = plt.subplots()
    ax.plot(x_plot, y_plot_smooth)
    ax.set_xlabel("Fluid Volumen")
    ax.set_ylabel("Predicted Probability")
    st.pyplot(fig)
    
    # Make 3D surface
    combinations = list(product(range_operation_time, range_fluid_sum))
    df_combinations = pd.DataFrame(combinations, columns=[ 'Operation time (min)', 'Fluid_sum'])
    
    df_repeated = pd.concat([df_input] * df_combinations.shape[0], ignore_index=True)
    
    # Concat df
    df_combinations = pd.concat([df_combinations.reset_index(drop = True),
                                 df_repeated.reset_index(drop = True)] , axis = 1)
    
    # Predict
    model.eval()
    with torch.no_grad():
        test_f1 = torch.tensor(df_combinations['Fluid_sum'].values).float().view(-1, 1)
        test_f2 = torch.tensor(df_combinations['Operation time (min)'].values).float().view(-1, 1)
        test_other = torch.tensor(df_combinations.drop(columns = ['Fluid_sum' , 'Operation time (min)']).values).float()

        test_output = model(test_f1, test_f2, test_other).squeeze()
    df_combinations['pred_proba'] = test_output.detach().numpy() * 100
    
    # Left only rows with pred proba between 0.2 and 0.8
    min_prob = np.min(df_combinations['pred_proba'].values)
    max_prob = np.max(df_combinations['pred_proba'].values)
    print('Min prob:' , min_prob , 'Max prob:' , max_prob)
    #df_combinations = df_combinations[(df_combinations['pred_proba'] >= 0.1)&(df_combinations['pred_proba'] <= 0.8)]
    print('Predictions generated')
    
    # Extract information for plot 3D
    df_plot = df_combinations[['Operation time (min)', 'Fluid_sum' , 'pred_proba']]
    
    # Find minimum risk
    min_row = df_plot.loc[df_plot['pred_proba'].idxmin()]
    min_X = min_row['Operation time (min)']
    min_Y = min_row['Fluid_sum']
    min_Z = min_row['pred_proba']
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Convert columns into matrix
    X_matrix = df_plot.pivot_table(index = 'Operation time (min)', columns = 'Fluid_sum', values =  'pred_proba').columns.values
    Y_matrix = df_plot.pivot_table(index = 'Operation time (min)', columns = 'Fluid_sum', values =  'pred_proba').index.values
    Z_matrix = df_plot.pivot_table(index = 'Operation time (min)', columns = 'Fluid_sum', values =  'pred_proba').values

    # Crear la superficie
    X_mesh, Y_mesh = np.meshgrid(X_matrix, Y_matrix)
    surf = ax.plot_surface(X_mesh, Y_mesh, Z_matrix, cmap = cm.coolwarm)

    # Colorbar
    fig.colorbar(surf, ax = ax, shrink = 0.5, aspect = 5)
    
    # Add a point in the plot with the minimum risk
    ax.scatter(min_X, min_Y, min_Z, color = 'red', s = 50, label = f"Mín (Operation Time = {min_X:.2f}, Fluid Volumen = {min_Y:.2f}, AL Likelihood = {min_Z:.2f})")
    ax.legend()

    # Axis labels
    ax.set_xlabel('Operation time (min)')
    ax.set_ylabel('Fluid_sum')
    ax.set_zlabel('Risk of Anastomotic Leakage')
    
    # Title
    ax.set_title('Predicted Anastomotic Leakage based on Surgery Time and Fluid Volumen')
    
    # Show Plot
    st.pyplot(fig)

    # Put a message with the minimum risk
    message = f"The minimum AL Likelihood is **{min_Z}**, which occurs with Operation Time = **{min_X}** and Fluid Volumen = **{min_Y}**"
    st.write(message)
    
    return None

###############################################################################
# Page configuration
st.set_page_config(
    page_title="AL Prediction App"
)
st.set_option('deprecation.showPyplotGlobalUse', False)
# Initialize app
model , preprocesor = initialize_app()

# Option Menu configuration
with st.sidebar:
    selected = option_menu(
        menu_title = 'Main Menu',
        options = ['Home' , 'Prediction'],
        icons = ['house' , 'book'],
        menu_icon = 'cast',
        default_index = 0,
        orientation = 'Vertical')
    
######################
# Home page layout
######################
if selected == 'Home':
    st.title('Anastomotic Leackage App')
    st.markdown("""
    This app contains 2 sections which you can access from the horizontal menu above.\n
    The sections are:\n
    Home: The main page of the app.\n
    **Prediction:** On this section you can select the patients information and
    the models iterate over all posible operation time and fluid volumen for suggesting
    the best option.\n
    """)
    
###############################################################################
# Prediction page layout
if selected == 'Prediction':
    st.title('Prediction Section')
    st.subheader("Description")
    st.subheader("To predict Anastomotic Leackage, you need to follow the steps below:")
    st.markdown("""
    1. Enter clinical parameters of patient on the left side bar.
    2. Enter initial values for operation time and fluid volumen.
    3. Press the "Predict" button and wait for the result.
    """)
    st.markdown("""
    This model predicts the probabilities of AL for simulated values of operation time and fluid volumen.
    """)
    
    # Sidebar layout
    st.sidebar.title("Patiens Info")
    st.sidebar.subheader("Please choose parameters")
    
    # Input features
    age = st.sidebar.slider("Age:", min_value = 18, max_value = 100,step = 1)
    bmi = st.sidebar.slider("Preoperative BMI:", min_value = 18, max_value = 50,step = 1)
    preoperative_hemoglobin_level = st.sidebar.slider("Preoperative Hemoglobin Level:", min_value = 0.0, max_value = 30.0,step = 0.1)
    preoperative_leukocyte_count_level = st.sidebar.slider("Preoperative Leukocyte Count:", min_value = 0.0, max_value = 30.0,step = 0.1)
    preoperative_albumin_level = st.sidebar.slider("Preoperative Albumin Level:", min_value = 0.0, max_value = 30.0,step = 0.1)
    preoperative_crp_level = st.sidebar.slider("Preoperative CRP Level:", min_value = 0.0, max_value = 100.0,step = 0.1)
    sex = st.sidebar.selectbox('Gender', tuple(INPUT_FEATURES['Sex (1: Male, 2: Female)'].keys()))
    active_smoking = st.sidebar.selectbox('Active Smoking', tuple(INPUT_FEATURES['Active Smoking (1: Yes, 0: No)'].keys()))
    alcohol_abuse = st.sidebar.selectbox('Alcohol Abuse', tuple(INPUT_FEATURES['Alcohol abuse (1: <2 beverages/day, 2: >= 2 beverages/day, 3: No alcohol abuse, 4:Unknown)'].keys()))
    renal_function = st.sidebar.selectbox('Renal Function CKD Stages', tuple(INPUT_FEATURES['Real function CKD stages G1 (normal) to G5 (1: G1, 2: G2, 3:G3a, 4: G3b, 5: G4, 6: G5)'].keys()))
    liver_metastasis = st.sidebar.selectbox('Liver Metastasis', tuple(INPUT_FEATURES['Liver metastasis at time of anastomosis (any) (1: Yes, 2: No, 3: Unknown)'].keys()))
    neoadjuvant_therapy = st.sidebar.selectbox('Neoadjuvant Therapy', tuple(INPUT_FEATURES['Neoadjuvant Therapy (1=yes, 0 = no)'].keys()))
    preoperative_use_immunodepressive_drugs = st.sidebar.selectbox('Use of Immunodepressive Drugs', tuple(INPUT_FEATURES['Preoperative use of immunosuppressive drugs 2 weeks before surgery (1: Yes, 0: No, 2: Unknown)'].keys()))
    preoperative_steroid_use = st.sidebar.selectbox('Steroid Use', tuple(INPUT_FEATURES[ 'Preoperative steroid use (1: Yes, 0: No, 2: Unknown)'].keys()))
    preoperative_nsaids_use = st.sidebar.selectbox('NSAIDs Use', tuple(INPUT_FEATURES['Preoperative NSAIDs use (1: Yes, 0: No, 2: Unknown)'].keys()))
    preoperative_blood_transfusion = st.sidebar.selectbox('Preoperative Blood Transfusion', tuple(INPUT_FEATURES['Preoperative blood transfusion (1: Yes, 0: No, 2: Unknown)'].keys()))
    tnf_alpha = st.sidebar.selectbox('TNF Alpha', tuple(INPUT_FEATURES['TNF Alpha Inhib (1=yes, 0=no)'].keys()))
    cci = st.sidebar.selectbox('Charlson Comorbility Index', tuple(INPUT_FEATURES['Charlson comorbidity index'].keys()))
    asa_score = st.sidebar.selectbox('ASA Score', tuple(INPUT_FEATURES['American Society of Anesthesiologists (ASA) Score (1: ASA 1: healthy person, 2: ASA 2: mild systemic disease, 3: ASA 3: severe systemic disease, 4: ASA 4: severe systemic disease that is a constant threat to life, 5: ASA 5: a moribund person who is not ex, 6: Unknown'].keys()))
    prior_abdominal_surgery = st.sidebar.selectbox('Prior abdominal surgery', tuple(INPUT_FEATURES['Prior abdominal surgery (1: Yes, 2: No, 3: Unknown)'].keys()))
    indication = st.sidebar.selectbox('Indication', tuple(INPUT_FEATURES['Indication (1: Recurrent Diverticulitis, 2: Acute Diverticulitis, 3: Ileus/Stenosis, 4: Ischemia, 5: Tumor, 6: Volvulus, 7: Morbus crohn, 8: Colitis ulcerosa, 9: Perforation (müsste perforation = yes und emergency = yes -> muss in 10 other), 10: Other, 11: Ileostoma reversal = zu 12, 12: Colostoma reversal'].keys()))
    operation_type = st.sidebar.selectbox('Operation', tuple(INPUT_FEATURES['Operation'].keys())) 
    emergency_surgery = st.sidebar.selectbox('Emergency Surgery', tuple(INPUT_FEATURES['Emergency surgery (1: Yes, 0: No, 2: Unknown)'].keys()))
    perforation = st.sidebar.selectbox('Perforation', tuple(INPUT_FEATURES['Perforation (1: Yes, 0: No)'].keys()))
    approach = st.sidebar.selectbox('Approach', tuple(INPUT_FEATURES['Approach (1: Laparoscopic, 2: Robotic, 3: Open, 4: Conversion to open, 5: Conversion to laparoscopy, 6: Transanal (ta TME, TATA, TAMIS))'].keys()))
    type_of_anastomosis = st.sidebar.selectbox('Type of Anastomosis', tuple(INPUT_FEATURES['Type of anastomosis (1: Colon anastomosis, 2: Colorectal anastomosis, 3: Ileocolonic anastomosis, 4: Ileorectal anastomosis, 5: Ileopouch-anal, 6: Colopouch, 7: Small intestinal anastomosis, 8: Unknown)'].keys()))
    anastomotic_technique = st.sidebar.selectbox('Anastomotic Technique', tuple(INPUT_FEATURES['Anastomotic technique (1: Stapler, 2: Hand-sewn, 3: Stapler and Hand-sewn, 4: Unknown) (alle 3 werden zu 1)'].keys()))
    anastomotic_configuration = st.sidebar.selectbox('Anastomotic Configuration', tuple(INPUT_FEATURES['Anastomotic configuration (1: end-to-end, 2: side-to-end, 3: side-to-side, 4: end-to-side, 5: Unknown)'].keys())) 
    protective_stomy = st.sidebar.selectbox('Protective Stomy', tuple(INPUT_FEATURES['Protective stomy (1: Ileostomy, 2: Colostomy, 3: No protective stomy, 4: Unknown)'].keys()))
    surgeon_experience = st.sidebar.selectbox('Surgeon Experience', tuple(INPUT_FEATURES[ "Surgeon's experience (1: Consultant (the counsalting performed the operation, the other persons only assisted), 2: Teaching operation (Consultant with senior resident, the Resident was allowed to do part or more of the case), 3: Unknown)"].keys()))
    total_points_nutritional_status = st.sidebar.selectbox('Points Nutritional Status', tuple(INPUT_FEATURES['Total points Nutritional status'].keys())) 
    
    
    # Add subheader for initial operation time and fluid volumen
    st.subheader("Initial Inputs for Fluid Volumen and Surgery Duration: ")
    operation_time = st.slider("Surgery Duration:" , min_value = 100.0, max_value = 600.0, step = 5.0)
    fluid_sum = st.slider("Fluid Volumen:" , min_value = 600.0, max_value = 200.0, step = 10.0)
    
    # Create df input
    df_input = pd.DataFrame({'Age (Years)' : [age],
                             'BMI' : [bmi],
                             'Preoperative hemoglobin level (in g/dL)' : [preoperative_hemoglobin_level],
                             'Preoperative leukocyte count (in 10^9/L)' : [preoperative_leukocyte_count_level],
                             'Preoperative albumin level (in g/dL)' : [preoperative_albumin_level],
                             'Preoperative CRP level (mg/l)' : [preoperative_crp_level],
                             'Sex (1: Male, 2: Female)' : [sex],
                             'Active Smoking (1: Yes, 0: No)' : [active_smoking],
                             'Alcohol abuse (1: <2 beverages/day, 2: >= 2 beverages/day, 3: No alcohol abuse, 4:Unknown)' : [alcohol_abuse],
                             'Real function CKD stages G1 (normal) to G5 (1: G1, 2: G2, 3:G3a, 4: G3b, 5: G4, 6: G5)' :[renal_function],
                             'Liver metastasis at time of anastomosis (any) (1: Yes, 2: No, 3: Unknown)' : [liver_metastasis],
                             'Neoadjuvant Therapy (1=yes, 0 = no)' : [neoadjuvant_therapy],
                             'Preoperative use of immunosuppressive drugs 2 weeks before surgery (1: Yes, 0: No, 2: Unknown)' : [preoperative_use_immunodepressive_drugs],
                             'Preoperative steroid use (1: Yes, 0: No, 2: Unknown)' : [preoperative_steroid_use],
                             'Preoperative NSAIDs use (1: Yes, 0: No, 2: Unknown)' : [preoperative_nsaids_use],
                             'Preoperative blood transfusion (1: Yes, 0: No, 2: Unknown)' : [preoperative_blood_transfusion],
                             'TNF Alpha Inhib (1=yes, 0=no)' : [tnf_alpha],
                             'Charlson comorbidity index' : [cci],
                             'American Society of Anesthesiologists (ASA) Score (1: ASA 1: healthy person, 2: ASA 2: mild systemic disease, 3: ASA 3: severe systemic disease, 4: ASA 4: severe systemic disease that is a constant threat to life, 5: ASA 5: a moribund person who is not ex, 6: Unknown' : [asa_score],
                             'Prior abdominal surgery (1: Yes, 2: No, 3: Unknown)' : [prior_abdominal_surgery],
                             'Indication (1: Recurrent Diverticulitis, 2: Acute Diverticulitis, 3: Ileus/Stenosis, 4: Ischemia, 5: Tumor, 6: Volvulus, 7: Morbus crohn, 8: Colitis ulcerosa, 9: Perforation (müsste perforation = yes und emergency = yes -> muss in 10 other), 10: Other, 11: Ileostoma reversal = zu 12, 12: Colostoma reversal' : [indication],
                             'Operation' : [operation_type],
                             'Emergency surgery (1: Yes, 0: No, 2: Unknown)' : [emergency_surgery],
                             'Perforation (1: Yes, 0: No)' : [perforation],
                             'Approach (1: Laparoscopic, 2: Robotic, 3: Open, 4: Conversion to open, 5: Conversion to laparoscopy, 6: Transanal (ta TME, TATA, TAMIS))' : [approach],
                             'Type of anastomosis (1: Colon anastomosis, 2: Colorectal anastomosis, 3: Ileocolonic anastomosis, 4: Ileorectal anastomosis, 5: Ileopouch-anal, 6: Colopouch, 7: Small intestinal anastomosis, 8: Unknown)' : [type_of_anastomosis],
                             'Anastomotic technique (1: Stapler, 2: Hand-sewn, 3: Stapler and Hand-sewn, 4: Unknown) (alle 3 werden zu 1)' : [anastomotic_technique],
                             'Anastomotic configuration (1: end-to-end, 2: side-to-end, 3: side-to-side, 4: end-to-side, 5: Unknown)' : [anastomotic_configuration],
                             'Protective stomy (1: Ileostomy, 2: Colostomy, 3: No protective stomy, 4: Unknown)' : [protective_stomy],
                             "Surgeon's experience (1: Consultant (the counsalting performed the operation, the other persons only assisted), 2: Teaching operation (Consultant with senior resident, the Resident was allowed to do part or more of the case), 3: Unknown)" : [surgeon_experience],
                             'Total points Nutritional status' : [total_points_nutritional_status]})
    # Parser input and make predictions
    predict_button = st.button('Predict')
    if predict_button:
        parser_input(df_input ,model , preprocesor , operation_time , fluid_sum)
