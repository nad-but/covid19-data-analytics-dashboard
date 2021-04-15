import streamlit as st
import pandas as pd
import altair as alt

st.cache(persist=True)
def load_data():
 url_conf='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
 covid_conf=pd.read_csv(url_conf)
 conf=covid_conf.melt(id_vars=["Province/State", "Country/Region","Lat","Long"], 
        var_name="Date", 
        value_name="Confirmed")
 conf["Date"] = pd.to_datetime(conf["Date"],format="%m/%d/%y")
 
 url_deaths="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
 covid_deaths=pd.read_csv(url_deaths)
 dea=covid_deaths.melt(id_vars=["Province/State", "Country/Region","Lat","Long"], 
        var_name="Date", 
        value_name="deaths")
 dea["Date"]=pd.to_datetime(dea["Date"],format="%m/%d/%y")
    
 url_recovered="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"
 covid_recovered=pd.read_csv(url_recovered)
 recov=covid_recovered.melt(id_vars=["Province/State", "Country/Region","Lat","Long"], 
        var_name="Date", 
        value_name="recovered")
 recov["Date"]=pd.to_datetime(recov["Date"],format="%m/%d/%y")

 covid19_DataFram=conf.join(dea['deaths'])
 return covid19_DataFram,recov
covid19_DataFram,recov=load_data()


st.sidebar.header("Covid-19 Data Analytics Dashbord") 
my_model= st.sidebar.radio('choose model:',('Information about COVID-19 pandemic','The Most Top 10 Country','Select Certain Country for Different Type of Cases','Time Series:Cumulative cases'))
st.sidebar.write('### Done by: Nadhira Albattashi and Buthaina ALsiyabi')


if my_model == 'Information about COVID-19 pandemic':
    st.write('# 🦠Covid-19 Data Analytics Dashbord🦠')

 

    st.write('## ⚠️ Information about COVID-19 pandemic⚠️')
    st.video('https://www.youtube.com/watch?v=U8r3oTVMtQ0')
    st.write('### ⭕ A detailed overview of the pandemic:⭕')
    st.write('Coronaviruses are a large family of enveloped, non-segmented, single-stranded, positive-sense RNA viruses that circulate among animals including camels, cats,  and bats. Coronaviruses derive their name from their electron microscopic image,which resembles a crown – or corona.Six strains of coronavirus have infected humans, four of which are together responsible for about one-third of common colds. In the past two decades, there have been three global coronavirus outbreaks')
    st.write('### ⭕ When it began and where it appeared:⭕ ')
    st.write('Retrospective investigations by Chinese authorities have identified human cases with onset of symptoms in early December 2019. While some of the earliest known cases had a link to a wholesale food market in Wuhan, some did not.')
    st.write('### ⭕Reason for its emergence:⭕')
    st.write('Anyone can get COVID-19, and most infections are mild. The older you are, the higher your risk of severe illness.Some children and teens who are in the hospital with COVID-19 have an inflammatory condition that doctors are calling multisystem inflammatory syndrome in children. Doctors think it may be linked to the virus. It causes symptoms similar to those of toxic shock and of Kawasaki disease, a condition that causes inflammation in kids’ blood vessels.')
    st.write('### ⭕ How the world has adopte to the existence of the pandemic ⭕ ')
    st.write("The world has become able to complete its life with the existence of the pandemic by committing to several steps, including wearing masks outside and in the workplace to avoid infection. Also perform work that is not required to come to the workplace at home. Avoid disruption as much as possible. Use sterilizers and hops to perform simpler activities such as shopping. Adopting the e-learning method instead of the attendance")
    st.write("### ⭕ Vaccine Types: ⭕")
    st.write("#### •    Understanding How COVID-19 Vaccines Work")
    st.write("Learn how the body fights infection and how COVID-19 vaccines protect people by producing immunity. Also see the different types of COVID-19 vaccines that currently are available or are undergoing large-scale (Phase 3) clinical trials in the United States.")
    st.write("#### •    COVID-19 mRNA Vaccines ")
    st.write("Information about mRNA vaccines generally and COVID-19 vaccines that use this new technology specifically.")
    st.write("#### •    Viral Vector COVID-19 Vaccines")
    st.write("Information about viral vector vaccines generally and COVID-19 vaccines that use this new technology specifically.")

elif my_model == 'The Most Top 10 Country':
    topconf  = covid19_DataFram.groupby('Country/Region').max().sort_values(by='Confirmed', ascending=False)[:10]
    topconf.reset_index(inplace=True)

    ch1conf = alt.Chart(topconf,width=600,height=400).encode(
      x=alt.X("Country/Region", sort='-y'),
      y="Confirmed",
      color="Country/Region",
      tooltip = "Confirmed"
     ).interactive()

    topdea = covid19_DataFram.groupby('Country/Region').max().sort_values(by='deaths', ascending=False)[:10]
    topdea.reset_index(inplace=True)


    ch2dea = alt.Chart(topdea,width=600,height=400).encode(
    x=alt.X("Country/Region", sort='-y'),
    y="deaths",
    color="Country/Region",
    tooltip = "deaths"
    ).interactive()



    toprecov  = recov.groupby('Country/Region').max().sort_values(by='recovered', ascending=False)[:10]
    toprecov.reset_index(inplace=True)


    ch3recov = alt.Chart(toprecov,width=600,height=400).encode(
      x=alt.X("Country/Region", sort='-y'),
      y="recovered",
      color="Country/Region",
      tooltip = "recovered"
     ).interactive()


    st.write("The Most: Top 10 countries")
    top10 = st.selectbox("Select your option",["Confirmed Cases","Deaths","Recovered"])
    typeofchart=['Bar Chart','Rule Chart','Circle Chart']
    selectchart=st.radio('select type of chart',typeofchart)
    if top10 == "Confirmed Cases":
       if selectchart == 'Bar Chart':
        st.altair_chart(ch1conf.mark_bar())
       elif selectchart == 'Rule Chart':
        st.altair_chart(ch1conf.mark_rule())
       else:
        st.altair_chart(ch1conf.mark_circle())
    
    elif top10 == "Deaths":
      if selectchart == 'Bar Chart':
        st.altair_chart(ch2dea.mark_bar())
      elif selectchart == 'Rule Chart':
        st.altair_chart(ch2dea.mark_rule())
      else:
        st.altair_chart(ch2dea.mark_circle())
    

    else:
     if selectchart == 'Bar Chart':
        st.altair_chart(ch3recov.mark_bar())
     elif selectchart == 'Rule Chart':
        st.altair_chart(ch3recov.mark_rule())
     else:
        st.altair_chart(ch3recov.mark_circle())

elif my_model == 'Select Certain Country for Different Type of Cases':
    
    st.write('## Select the Country to display Chart')
    
    select_country = st.selectbox("Select country: ",covid19_DataFram["Country/Region"].unique())
    option = ['Infection','Deaths','Recovered']
    select_option =st.radio('Select your option', option)
    selchart =st.selectbox('Select type of Chart', ['Line Chart','Area Chart'])

    infication= alt.Chart(covid19_DataFram[covid19_DataFram["Country/Region"]== select_country]).encode(
    x="month(Date)",
    y=alt.Y("Confirmed",title="infection cases"),
    column="year(Date)" ,
    tooltip=['sum(Confirmed)']
    ).interactive()

    death= alt.Chart(covid19_DataFram[covid19_DataFram["Country/Region"]== select_country]).encode(
     x="month(Date)",
     y=alt.Y("deaths",title="Deaths cases"),
     column="year(Date)",
     tooltip=['sum(deaths)']
    ).interactive()

    Recovered= alt.Chart(recov[recov["Country/Region"]== select_country]).encode(
    x="month(Date)",
    y=alt.Y("recovered",title="Recovered cases"),
    column="year(Date)",
    tooltip=['sum(recovered)']
    ).interactive()

    if select_option == 'Infection':
      if selchart == 'Line Chart':
        st.altair_chart(infication.mark_line(color='green'))
      else:
        st.altair_chart(infication.mark_area(color='red'))

    elif select_option == 'Deaths':
      if selchart == 'Line Chart':
        st.altair_chart(death.mark_line(color='green'))
      else:
        st.altair_chart(death.mark_area(color='red'))

    else:
        if selchart == 'Line Chart':
          st.altair_chart(Recovered.mark_line(color='green'))
        else:
          st.altair_chart(Recovered.mark_area(color='red'))



else:
    st.write('## Time Series:Cumulative infection cases /deaths/recovered through months of the year')


    times=['Cumulative infection cases through months of the year','Cumulative deaths through months of the year','Cumulative recovered through months of the year']

    timeseries=st.selectbox('Select your option', times)

    typeofchart=['Bar Chart','Point Chart']
    selectchart=st.radio('select type of chart',typeofchart)




    infectionCases=alt.Chart(covid19_DataFram).encode(
    x="month(Date)",
    y=alt.Y("sum(Confirmed)",title="Cumulative infection cases"),
    tooltip=['sum(Confirmed)'],
    column="year(Date)"
    ).interactive()

    death=alt.Chart(covid19_DataFram).encode(
    x="month(Date)",
    y=alt.Y("sum(deaths)",title="Cumulative deaths"),
    tooltip=['sum(deaths)'],
    column="year(Date)"
    ).interactive()

    recovered=alt.Chart(covid19_DataFram).encode(
    x="month(Date)",
    y=alt.Y("sum(recovered)",title="Cumulative recovered"),
    tooltip=['sum(recovered)'],
    column="year(Date)"
    ).interactive()

    if timeseries == 'Cumulative infection cases through months of the year':
      if selectchart == 'Bar Chart':
        st.altair_chart(infectionCases.mark_bar())
      else:
        st.altair_chart(infectionCases.mark_point(color='red'))

    elif timeseries == 'Cumulative deaths through months of the year':
       if selectchart == 'Bar Chart':
         st.altair_chart(death.mark_bar())
       else:
        st.altair_chart(death.mark_point(color='red'))

    else:
      if selectchart == 'Bar Chart':
        st.altair_chart(recovered.mark_bar())
      else:
        st.altair_chart(recovered.mark_point(color='red'))


    

