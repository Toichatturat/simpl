import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import warnings
import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import base64
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor







st.set_page_config(page_title="Dashboard By Natthaphat", page_icon=":bar_chart:",layout="wide")
st.title(" :bar_chart: Salary Estimate")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)
box_date = str(datetime.datetime.now().strftime("%d %B %Y"))
st.write(f"Last updated by:  \n {box_date}")

# st.cache(clear_cache=True)
age = (
    '18-24','25-34','35-44','45-54','55-64','65 or over','under 18',
)

job = (
    'Director', 'Executive Assistant', 'Librarian' ,'Manager' ,'Other',
    'Program Manager', 'Project Manager', 'Senior Software Engineer',
    'Software Engineer' ,'Teacher'
)

state = (
    'California' ,'Colorado' ,'District of Columbia' ,'Florida', 'Georgia',
    'Illinois', 'Maryland' ,'Massachusetts', 'Michigan' ,'Minnesota','New York',
    'North Carolina', 'Ohio' ,'Oregon', 'Pennsylvania', 'Texas', 'Virginia',
    'Washington', 'Wisconsin'
)

experience = (
    '1 year or less', '11 - 20 years', '2 - 4 years' ,'21 - 30 years',
    '31 - 40 years' ,'41 years or more', '5-7 years', '8 - 10 years'
)

education = (
    "Bachelor’s degree" ,"Less than a Bachelors", "Master’s degree", "Post grad"
)

gender = (
    'Man' ,'Woman'
)

industry = (
    'Education (Higher Education)', 'Nonprofits',
    'Accounting, Banking & Finance', 'Business or Consulting', 'Other',
    'Health care', 'Government and Public Administration', 'Law',
    'Engineering or Manufacturing', 'Media & Digital',
    'Computing or Tech', 'Education (Primary/Secondary)',
    'Marketing, Advertising & PR', 'Insurance'
)



job_input = st.selectbox("Job Title", job)
industry_input = st.selectbox("Industry", industry)
in1,in2 = st.columns((2))
with in1:
    age_input = st.selectbox("Age", age)
with in2:
    experience_input = st.selectbox("Experience", experience)
state_input = st.selectbox("State", state)
in3,in4 = st.columns((2))
with in3:
    education_input = st.selectbox("Education", education)
with in4:
    gender_input = st.selectbox("Gender", gender)

# Placeholder value
placeholder_value = 200000

# Input เป็นตัวเลข
salary_input = st.number_input("Enter Salary", value=placeholder_value, step=1000)

# ตรวจสอบว่าผู้ใช้ได้ป้อนค่าหรือไม่
if salary_input == placeholder_value:
    st.warning("Please enter a salary.")
else:
    st.success(f"You entered: {salary_input}")



def load_model():
    with open('lr_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data
def load_model_rf():
    with open('rf_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()
rf = load_model_rf()

rf_model = rf['model']
regressor_loaded = data["model"]
le_age = data["le_age"]
le_job = data["le_job"]
le_state = data["le_state"]
le_experience = data["le_experience"]
le_education = data["le_education"]
le_gender = data["le_gender"]
le_industry = data["le_industry"]


def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map

def clean_experience(x):
    if x ==  'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)


def clean_education(x):
    if 'College degree' in x:
        return 'Bachelor’s degree'
    if "Master's degree" in x:
        return 'Master’s degree'
    if 'Professional degree (MD, JD, etc.)' in x or 'PhD' in x:
        return 'Post grad'
    return 'Less than a Bachelors'




# @st.cache
def load_data():
    df = pd.read_csv("real-salary.csv")
    df = df.rename(columns={"How old are you?": "age", "Job title": "job", "What is your annual salary? (You'll indicate the currency in a later question. If you are part-time or hourly, please enter an annualized equivalent -- what you would earn if you worked the job 40 hours a week, 52 weeks a year.)": "salary","If you're in the U.S., what state do you work in?":"state","What city do you work in?":"city","How many years of professional work experience do you have in your field?":"experience","What is your highest level of education completed?":"education","What is your gender?":"gender","What industry do you work in?":"industry"})
    df = df[["age", "job", "salary", "state","city", "experience", "education", "gender","industry"]]
    df = df[["age", "job", "salary", "state","city", "experience", "education", "gender","industry"]].dropna()
    df = df.dropna()
    df['salary'] = df['salary'].replace('[\$,]', '', regex=True).astype(float)
    df = df.drop(columns=['city'])

    country_map = shorten_categories(df.state.value_counts(), 400)
    df["state"] = df["state"].map(country_map)
    df = df[df["salary"] <= 250000]
    df = df[df["salary"] >= 10000]
    df = df[df["state"] != "Other"]
    df = df[df['gender'] != 'Other or prefer not to answer']
    df = df[df['gender'] != 'Non-binary']
    job_map = shorten_categories(df.job.value_counts(), 100)
    df["job"] = df["job"].map(job_map)
    industry_map = shorten_categories(df.industry.value_counts(), 400)
    df["industry"] = df["industry"].map(industry_map)

    df['education'] = df['education'].apply(clean_education)
    return df

df = load_data()

# age_input = age
# job_input = job
# state_input = state
# experience_input = experience
# education_input =  education
# gender_input = gender
# industry_input = industry


col1, col2, col3 = st.columns((3))
with col1:
    estimate = st.button("Salary Estimate")
with col2:
    estimate2 = st.button("Predict Salary")
with col3:
    estimate3 = st.button("Analysis Model")
        
if estimate: 
    st.title("Salary Estimate")

    st.write(
        """
    ### Real Survey from interviewing Survey.
    """
    )
    st.write(
        """
    Data Salary to Estimate
    """
    )
    

    salary_summary = df['salary'].describe()
    # st.write(salary_summary, use_container_width=True)
    st.write(f"Your Salary to Estimate is ${salary_input} USD " )
        # กรอง DataFrame เฉพาะงานที่ input มา
    filtered_df = df[df['job'] == job_input]

    # สร้างกราฟ Boxplot ด้วย plotly
    fig = px.box(filtered_df, x='job', y='salary', points='all', title=f'Salary Distribution for {job_input}')
    fig.add_shape(type='line', x0=0, x1=1, y0=salary_input, y1=salary_input,
                line=dict(color='red', width=2, dash='dash'),
                name=f'Input Salary ({salary_input})')

    fig.add_shape(type='line', x0=0, x1=1, y0=filtered_df['salary'].mean(), y1=filtered_df['salary'].mean(),
                line=dict(color='green', width=2, dash='dash'),
                name=f'Mean Salary ({filtered_df["salary"].mean():.2f})')
    # เพิ่ม annotation ของ Your Salary
    fig.add_annotation(x=0.5, y=salary_input,
                text='Your Salary',
                showarrow=True,
                arrowhead=2,
                ax=20, ay=-40,
                font=dict(size=12, color='white'),
                bgcolor='red',
                opacity=0.8)
    fig.add_annotation(x=0.8, y=filtered_df['salary'].mean(),
                text='Average of Salary',
                showarrow=True,
                arrowhead=2,
                ax=-30, ay=35,
                font=dict(size=12, color='white'),
                bgcolor='blue',
                opacity=0.8)
    c1 , c2 = st.columns((2))
    with c1:
        feature_importance = rf_model.feature_importances_
        # สร้างกราฟ Feature Importance ด้วย Plotly
        fig2 = go.Figure()

        fig2.add_trace(go.Bar(
            y=np.arange(1, 11),
            x=feature_importance,
            orientation='h',
            marker=dict(color=feature_importance, colorscale='Viridis', reversescale=True),
            text=np.round(feature_importance, 3),
            textposition='auto',
        ))

        fig2.update_layout(
            title="Feature Importance for Salary",
            xaxis_title="Importance",
            yaxis_title="Feature",
            template="plotly_white",
        )

        # แสดงกราฟใน Streamlit
        st.plotly_chart(fig2, use_container_width=True)
        st.write("Feature: 1 = Age, 2 = Job, 3 = State, 4 = Experience, 5 = Education, 6 = Gender, 7 = Industry")
    with c2:
        st.plotly_chart(fig, use_container_width=True)
    

    df_age = df[['age', 'salary', 'job']]
    filtered_df = df_age[df_age['job'] == job_input]
    
    

    # หาค่าเฉลี่ยของเงินเดือนสำหรับแต่ละกลุ่มอายุ
    mean_salary_by_age = filtered_df.groupby('age')['salary'].mean().reset_index()

    # สร้างกราฟเส้น
    fig = px.line(mean_salary_by_age, x='age', y='salary', markers=True, line_shape='linear')
    fig.update_layout(title=f'Mean Salary by Age for {job_input}', xaxis_title='Age', yaxis_title='Mean Salary', template='plotly')
    
    # เพิ่ม Scatter plot เพื่อแสดงลูกศร
    fig.add_trace(go.Scatter(x=mean_salary_by_age['age'], y=mean_salary_by_age['salary'],
                        mode='markers+text',
                        marker=dict(color='red', size=8),
                        text=mean_salary_by_age['salary'].round(2),
                        textposition='bottom center'))
    # สร้างปุ่มใน Streamlit
    # สร้าง state สำหรับการควบคุมการแสดงผลของปุ่ม
#     show_input_point = st.checkbox("Show Input Point on Graph")

# # ตรวจสอบว่าปุ่มถูกกดหรือไม่
#     if show_input_point:
#     # เพิ่มจุดสีฟ้าบนกราฟ
#         input_point_trace = go.Scatter(x=[age_input], y=[salary_input],
#                                     mode='markers',
#                                     marker=dict(color='blue', size=10, symbol='star'),
#                                     name='Input Point')
#         fig.add_trace(input_point_trace)
    # เพิ่มจุดสีฟ้าบนกราฟ
    fig.add_trace(go.Scatter(x=[age_input], y=[salary_input],
                        mode='markers',
                        marker=dict(color='blue', size=10, symbol='star'),
                        name='Input Point'))
    fig.add_annotation(x=age_input, y=salary_input,
                text='Your Salary',
                showarrow=True,
                arrowhead=2,
                ax=-10, ay=-30,
                font=dict(size=9, color='white'),
                bgcolor='red',
                opacity=0.6)
    # แสดงกราฟใน Streamlit
    st.plotly_chart(fig, use_container_width=True)
    
    df_state = df[['state', 'salary', 'job']]
    filtered_df = df_state[df_state['job'] == job_input]

    # หาค่าเฉลี่ยของเงินเดือนสำหรับแต่ละกลุ่มอายุ
    mean_salary_by_state = filtered_df.groupby('state')['salary'].mean().reset_index()

    # สร้างกราฟเส้น
    fig = px.line(mean_salary_by_state, x='state', y='salary', markers=True, line_shape='linear')
    fig.update_layout(title=f'Mean Salary by State for {job_input}', xaxis_title='State', yaxis_title='Mean Salary', template='plotly')

    # เพิ่ม Scatter plot เพื่อแสดงลูกศร
    fig.add_trace(go.Scatter(x=mean_salary_by_state['state'], y=mean_salary_by_state['salary'],
                            mode='markers+text',
                            marker=dict(color='red', size=8),
                            text=mean_salary_by_state['salary'].round(2),
                            textposition='bottom center'))

    # เพิ่มจุด Input Point
    fig.add_trace(go.Scatter(x=[state_input], y=[salary_input],
                            mode='markers',
                            marker=dict(color='blue', size=10, symbol='circle'),
                            name='Your input'))

    # เพิ่มเส้นตัดแกน Y
    fig.add_shape(type='line', x0=state_input, x1=max(mean_salary_by_state['state']), y0=salary_input, y1=salary_input,
                    line=dict(color='red', width=2, dash='dash'),
                    name=f'Your Input - Salary ({salary_input})')
    fig.add_shape(type='line', x0=0, x1=state_input, y0=salary_input, y1=salary_input,
                line=dict(color='red', width=2, dash='dash'),
                name=f'Your Input - Salary ({salary_input})')
    fig.add_annotation(x=state_input, y=salary_input,
                text='Your Salary',
                showarrow=True,
                arrowhead=2,
                ax=20, ay=-25,
                font=dict(size=9, color='white'),
                bgcolor='red',
                opacity=0.6)

    st.plotly_chart(fig, use_container_width=True)

    
    edu_df = df[['age', 'salary', 'education','job']]
    filtered_df = edu_df[edu_df['job'] == job_input]
    ed_df = filtered_df.groupby(by=["education", "age"], as_index=False)["salary"].mean()

    fig = go.Figure()

    for education in ed_df['education'].unique():
        education_data = ed_df[ed_df['education'] == education]
        fig.add_trace(go.Scatter(x=education_data['age'], y=education_data['salary'],
                                mode='lines+markers', name=education))
        # fig.add_trace(go.Scatter(x=education_data['age'], y=education_data['salary'],
        #                     mode='markers+text',
        #                     marker=dict(color='white', size=8),
        #                     text=education_data['salary'].round(2),
        #                     textposition='bottom center'))

    fig.update_layout(title= f'Salary for Age each Education by {job_input} ',
                    xaxis_title='Age',
                    yaxis_title='Salary',
                    legend_title='Education')
    ed1, gen1 = st.columns([0.7,0.3])
    
    with ed1:
        st.plotly_chart(fig, use_container_width=True)
    with gen1:
        df_gen = df[['job','salary','gender']]
        filtered_df = df_gen[df_gen['job'] == job_input]
        mean_salary_by_gen = filtered_df.groupby('gender')['salary'].mean().reset_index()
        mean_salary_by_gen['salary'] = mean_salary_by_gen['salary'].replace('[\$,]', '', regex=True).astype(int)
        fig = px.bar(mean_salary_by_gen, x='gender', y='salary', 
                    color='gender', 
                    text='salary',
                    title=f'Mean Salary by Gender for {job_input}',
                    color_discrete_sequence=px.colors.qualitative.Plotly)  # สีที่ใช้

        fig.update_layout(xaxis_title='Gender', yaxis_title='Salary')
        
        # เพิ่มค่าเฉลี่ยที่แต่ละแท่ง
        fig.update_traces(textposition='outside', hovertemplate='%{customdata[1]:,.0f}',)
        fig.update_traces(textposition='outside',)
        # เพิ่มจุด Input Point
        fig.add_trace(go.Scatter(x=[gender_input], y=[salary_input],
                                mode='markers',
                                marker=dict(color='blue', size=10, symbol='circle'),
                                name='Your input'))

        # เพิ่มเส้นตัดแกน Y
        fig.add_shape(type='line', x0=gender_input, x1=max(mean_salary_by_gen['gender']), y0=salary_input, y1=salary_input,
                        line=dict(color='red', width=2, dash='dash'),
                        name=f'Your Input - Salary ({salary_input})')
        fig.add_shape(type='line', x0=0, x1=gender_input, y0=salary_input, y1=salary_input,
                    line=dict(color='red', width=2, dash='dash'),
                    name=f'Your Input - Salary ({salary_input})')
        fig.add_annotation(x=gender_input, y=salary_input,
                text='Your Salary',
                showarrow=True,
                arrowhead=3,
                ax=-15, ay=-25,
                font=dict(size=9, color='white'),
                bgcolor='red',
                opacity=0.6)

        st.plotly_chart(fig, use_container_width=True)

    # df_weekly_sales = df[['Week', 'Price']]

    # df_weekly_sales = df_weekly_sales.groupby('Week')['Price'].sum().reset_index()

    # st.header("Total Weekly sales")

    # fig, axes = plt.subplots(2, 1, figsize=(12, 16))
    # sns.barplot(x='Week', y='Price', data=df_weekly_sales, palette='viridis', ax=axes[0])
    # axes[0].set_title('Total Weekly Sales (Barplot)')
    # axes[0].set_xlabel('Week')
    # axes[0].set_ylabel('Total Sales')
    # sns.lineplot(x='Week', y='Price', data=df_weekly_sales, marker='o', ax=axes[1])
    # axes[1].set_title('Total Weekly Sales (Lineplot)')
    # axes[1].set_xlabel('Week')
    # axes[1].set_ylabel('Total Sales')
    # plotly_fig = make_subplots(rows=1, cols=2)
    # plotly_fig.add_trace(go.Bar(x=df_weekly_sales['Week'], y=df_weekly_sales['Price'], name='Bar Chart'), row=1, col=2)
    # plotly_fig.add_trace(go.Scatter(x=df_weekly_sales['Week'], y=df_weekly_sales['Price'], mode='lines+markers', name='Line Chart'), row=1, col=1)
    # st.plotly_chart(plotly_fig,use_container_width=True)
    # _, view1, dwn1, _ = st.columns((4))
    # with view1:
    #     expander = st.expander("Weekly sales Report")
    #     data = df[["Week","Price"]].groupby(by="Week")["Price"].sum()
    #     expander.write(data)
    # with dwn1:
    #     st.download_button("Get Data", data = data.to_csv().encode("utf-8"), 
    #                     file_name="Weekly-sales.csv", mime="text/csv")
    
    # สร้างปุ่มใน Streamlit
    # สร้าง state สำหรับการควบคุมการแสดงผลของปุ่ม
#     show_input_point = st.checkbox("Show Input Point on Graph")

# # ตรวจสอบว่าปุ่มถูกกดหรือไม่
#     if show_input_point:
#     # เพิ่มจุดสีฟ้าบนกราฟ
#         input_point_trace = go.Scatter(x=[age_input], y=[salary_input],
#                                     mode='markers',
#                                     marker=dict(color='blue', size=10, symbol='star'),
#                                     name='Input Point')
#         fig.add_trace(input_point_trace)
    # เพิ่มจุดสีฟ้าบนกราฟ
    
    # แสดงกราฟใน Streamlit
    
    df_industry = df[['job','salary','industry']]
    filtered_df = df_industry[df_industry['job'] == job_input]
    mean_salary_by_industry = filtered_df.groupby('industry')['salary'].mean().reset_index()
    mean_salary_by_industry['salary'] = mean_salary_by_industry['salary'].replace('[\$,]', '', regex=True).astype(int)
    fig = px.bar(mean_salary_by_industry, x='industry', y='salary', 
                color='industry', 
                custom_data=['industry', 'salary'],
                text='salary',
                title=f'Mean Salary by Age for {job_input}',
                color_discrete_sequence=px.colors.qualitative.Plotly)  # สีที่ใช้

    fig.update_layout(xaxis_title='industry', yaxis_title='Salary')
    # เพิ่มค่า mean salary ในแต่ละแท่ง
    # เพิ่มลูกศรชี้บอกค่านั้น ๆ บนแท่ง
    fig.update_traces(textposition='outside', hovertemplate='%{customdata[1]:,.0f}',)
    fig.update_traces(textposition='outside',)
    
    # เพิ่มจุดสีฟ้าบนกราฟ
    fig.add_trace(go.Scatter(x=[industry_input], y=[salary_input],
                    mode='markers',
                    marker=dict(color='green', size=10, symbol='circle'),
                    name='Your Input',
                    text=["Your Input: Industry - {}, Salary - ${}".format(industry_input, salary_input)],
                    hoverinfo='text'))
    # เพิ่มเส้นลากตัดแกน y
    # เพิ่มเส้นตัดแกน x
    fig.add_shape(type='line', x0=industry_input, x1=max(mean_salary_by_industry['industry']), y0=salary_input, y1=salary_input,
                line=dict(color='red', width=2, dash='dash'),
                name=f'Your Input - Salary ({salary_input})')
    fig.add_shape(type='line', x0=0, x1=industry_input, y0=salary_input, y1=salary_input,
                line=dict(color='red', width=2, dash='dash'),
                name=f'Your Input - Salary ({salary_input})')

    # fig.update_traces(text=[f"Your Input: Industry - {industry_input}, Salary - ${salary_input}"], textposition='outside')
    fig.add_annotation(x=industry_input, y=salary_input,
                text='Your Salary',
                showarrow=True,
                arrowhead=4,
                ax=20, ay=-40,
                font=dict(size=9, color='white'),
                bgcolor='red',
                opacity=0.6)



    st.plotly_chart(fig, use_container_width=True)
        
if estimate2: 
    st.title(f"This part you can predict the Salary you shoule earn.")
    st.subheader(f"You can use this number to negotiate Salary")
    X = np.array([[age_input, job_input, state_input,experience_input,education_input, gender_input, industry_input]])
    X[:, 0] = le_age.transform(X[:,0])
    X[:, 1] = le_job.transform(X[:,1])
    X[:, 2] = le_state.transform(X[:,2])
    X[:, 3] = le_experience.transform(X[:,3])
    X[:, 4] = le_education.transform(X[:,4])
    X[:, 5] = le_gender.transform(X[:,5])
    X[:, 6] = le_industry.transform(X[:,6])
    X = X.astype(float)

    salary = regressor_loaded.predict(X)
    st.subheader(f"The estimated salary is ${salary[0]:.2f} USD")
        
    feature_importance = rf_model.feature_importances_
    # สร้างกราฟ Feature Importance ด้วย Plotly
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=np.arange(1, 11),
        x=feature_importance,
        orientation='h',
        marker=dict(color=feature_importance, colorscale='Viridis', reversescale=True),
        text=np.round(feature_importance, 3),
        textposition='auto',
    ))

    fig.update_layout(
        title="Feature Importance for Salary",
        xaxis_title="Importance",
        yaxis_title="Feature",
        template="plotly_white",
    )

    # แสดงกราฟใน Streamlit
    st.plotly_chart(fig, use_container_width=True)
    coefficients = regressor_loaded.coef_
    st.write(coefficients)
        
if estimate3: 
    st.title(f"This part examines the selection and performance of wage prediction models.")
    df2 = df.copy()
    
    le_education = LabelEncoder()
    df2['education'] = le_education.fit_transform(df2['education'])

    le_state = LabelEncoder()
    df2['state'] = le_state.fit_transform(df2['state'])
    
    le_experience = LabelEncoder()
    df2['experience'] = le_experience.fit_transform(df2['experience'])
    
    le_job = LabelEncoder()
    df2['job'] = le_job.fit_transform(df2['job'])
    
    le_age = LabelEncoder()
    df2['age'] = le_age.fit_transform(df2['age'])
    
    le_gender = LabelEncoder()
    df2['gender'] = le_gender.fit_transform(df2['gender'])
    
    le_industry = LabelEncoder()
    df2['industry'] = le_industry.fit_transform(df2['industry'])
    
    X = df2.drop("salary", axis=1)
    y = df2["salary"]
    
    
    linear_reg = LinearRegression()
    linear_reg.fit(X, y.values)
    
    y_pred = linear_reg.predict(X)
    


    # เพิ่มคอลัมน์ค่าคงที่ใน DataFrame
    X = sm.add_constant(X)

    # สร้างแบบจำลอง
    model = sm.OLS(y, X).fit()

    # แสดงข้อมูลทางสถิติ
    
    
    st.write("#### Mean Squared Error(MSE) is used to determine which model should be used for prediction. Here we will use 3 test models: LinearRegression, DecisionTreeRegressor and RandomForestRegressor. By the value of Mean squared error, whichever value is highest, that value should be used.")
    error = np.sqrt(mean_squared_error(y, y_pred))
    
    
    X = X.drop(columns = ["const"])
    
    dec_tree_reg = DecisionTreeRegressor(random_state=0)
    dec_tree_reg.fit(X, y.values)
    y_pred_tree = dec_tree_reg.predict(X)
    error2 = np.sqrt(mean_squared_error(y, y_pred_tree))
    
    
    
    random_forest_reg = RandomForestRegressor(random_state=0)
    random_forest_reg.fit(X, y.values)
    y_pred_random = random_forest_reg.predict(X)
    error3 = np.sqrt(mean_squared_error(y, y_pred_random))
    
    
    er1, er2, er3 = st.columns(3)
    with er1:
        st.write(f"MSE of LinearRegression is {error}")
    with er2:
        st.write(f"MSE of DecisionTreeRegressor is {error2}")
    with er3:
        st.write(f"MSE of RandomForestRegressor is {error3}")
        
    st.write(model.summary())
    
    st.write("A simple explanation is to look at the value P>|t| of each variable. If this value is greater than 0.05, you should eliminate that variable.")
    
    st.subheader("GridSearchCV")
    st.write("GridSearchCV is a function that is used to find values. hyperparameters that will be used to tune our model to the best settings.")

    feature_importance = rf_model.feature_importances_
    # สร้างกราฟ Feature Importance ด้วย Plotly
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=np.arange(1, 11),
        x=feature_importance,
        orientation='h',
        marker=dict(color=feature_importance, colorscale='Viridis', reversescale=True),
        text=np.round(feature_importance, 3),
        textposition='auto',
    ))

    fig.update_layout(
        title="Feature Importance for Salary",
        xaxis_title="Importance",
        yaxis_title="Feature",
        template="plotly_white",
    )

    # แสดงกราฟใน Streamlit
    st.plotly_chart(fig, use_container_width=True)
    coefficients = regressor_loaded.coef_
    st.subheader("Coefficients")
    st.write(coefficients)
    st.write("Features Importance is a value that indicates what is important to the salary rate that will be received.")
    st.write("Coefficients are values that indicate in which way each thing affects salary. The higher the value, the more positive the effect, or the more negative the value, indicating that the salary will be in strong opposition to that value.")
