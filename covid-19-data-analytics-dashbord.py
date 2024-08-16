import streamlit as st
import pandas as pd
import altair as alt
import folium
from streamlit_folium import folium_static

#from pycaret.internal.pycaret_experiment import TimeSeriesExperiment

from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch
import plotly.express as px
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
 newrecov=recov.groupby(['Country/Region','Date'])["recovered"]
 newrecov=newrecov.sum().diff().reset_index()
 shift= newrecov['Country/Region'] != newrecov['Country/Region'].shift(1)
 import numpy as np
 newrecov.loc[shift,"recovered"]=np.nan
 newrecov.columns=['Country/Region','Date',"New recovered"]
 recov = pd.merge(recov, newrecov, on=['Country/Region', 'Date'])
 recov["New recovered"]=recov["New recovered"].fillna(0)
 recov = recov.replace({'Country/Region' : 'US'}, 'United States')
 oldcovid19_DataFram=conf.join(dea['deaths'])
 oldcovid19_DataFram = oldcovid19_DataFram.replace({'Country/Region' : 'US'}, 'United States')
 covid19_DataFram = oldcovid19_DataFram.groupby(['Date', 'Country/Region'])['Confirmed', 'deaths'].sum().reset_index()
 newcases=covid19_DataFram.groupby(['Country/Region','Date'])['Confirmed','deaths']
 newcases=newcases.sum().diff().reset_index()
 shift= newcases['Country/Region'] != newcases['Country/Region'].shift(1)
 newcases.loc[shift,'Confirmed']=np.nan
 newcases.loc[shift,'deaths']=np.nan
 newcases.columns=['Country/Region','Date','New Cases','New Deaths']
 covid19_DataFram = pd.merge(covid19_DataFram, newcases, on=['Country/Region', 'Date'])
 covid19_DataFram[['New Cases','New Deaths']]=covid19_DataFram[['New Cases','New Deaths']].fillna(0)
 
 
 url_vaccin='https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations.csv'
 covid_vaccin=pd.read_csv(url_vaccin)
 covid_vaccin['date']= pd.to_datetime(covid_vaccin['date'],format="%Y-%m-%d")
 covid_vaccin=covid_vaccin.fillna(0)
 
 #popul='https://raw.githubusercontent.com/owid/covid-19-data/master/scripts/input/un/population_2020.csv'
 #population=pd.read_csv(popul)
 popul='https://raw.githubusercontent.com/owid/covid-19-data/master/scripts/input/un/population_latest.csv'
 population=pd.read_csv(popul)
 #population=pd.read_csv("C:/Users/LENOVO/.spyder-py3/project/population_by_country_2020.csv")
 
 population=population.rename(columns={'entity': 'Country/Region'})
 population.drop(['iso_code', 'year','source'], axis=1, inplace=True)
 covid19_DataFram=covid19_DataFram.merge(population, on='Country/Region', how='left')
 covid19_DataFram['population']=covid19_DataFram['population'].fillna(0)
 
 covid19_DataFram['deaths percentage']=covid19_DataFram['deaths']/covid19_DataFram['Confirmed']*100
 covid19_DataFram['confirmed percentage']=covid19_DataFram['Confirmed']/covid19_DataFram['population']*100
 covid19_DataFram[['deaths percentage',	'confirmed percentage']]=covid19_DataFram[['deaths percentage',	'confirmed percentage']].fillna(0)
 covid19_DataFram[['deaths percentage','confirmed percentage']]=covid19_DataFram[['deaths percentage','confirmed percentage']].round(2)
 
 oldcovid19_DataFram=oldcovid19_DataFram.merge(population, on='Country/Region', how='left')
 oldcovid19_DataFram['population']=oldcovid19_DataFram['population'].fillna(0)
 oldcovid19_DataFram['deaths percentage']=oldcovid19_DataFram['deaths']/oldcovid19_DataFram['Confirmed']*100
 oldcovid19_DataFram['confirmed percentage']=oldcovid19_DataFram['Confirmed']/oldcovid19_DataFram['population']*100
 oldcovid19_DataFram[['deaths percentage',	'confirmed percentage']]=oldcovid19_DataFram[['deaths percentage',	'confirmed percentage']].fillna(0)
 oldcovid19_DataFram[['deaths percentage','confirmed percentage']]=oldcovid19_DataFram[['deaths percentage','confirmed percentage']].round(2)

 
 return oldcovid19_DataFram,covid19_DataFram,recov,covid_vaccin,population
oldcovid19_DataFram,covid19_DataFram,recov,covid_vaccin,population=load_data()


st.sidebar.header("Covid-19 Data Analytics Dashbord") 
my_model= st.sidebar.radio('choose model:',('World Map','Information about COVID-19 pandemic','The Most','Country wise analysis','Cumulative Data Visuals',"Oman Cases",'Compare Countries','Vaccination',"Forecasting","Search",'Help'))
st.sidebar.write('### Done by: Nadhira Albattashi and Buthaina ALsiyabi')

if my_model == "World Map":
    st.markdown("<h2 style='text-align: center;'>🦠Covid-19 Data Analytics Dashbord🦠</h2>", unsafe_allow_html=True) 
    totalvacc=covid_vaccin[covid_vaccin['location']=='World']

    groubdate=covid19_DataFram.groupby('Date').sum()
    groubdate.reset_index(inplace=True)

    totalconfirmed=groubdate['Confirmed'].tail(1).to_string(index=False)
    import re
    totalconfirme=re.sub(r'(\d{3})(?=\d)', r'\1,', str(totalconfirmed)[::-1])[::-1]
    
    totaldeath=groubdate['deaths'].tail(1).to_string(index=False)
    totaldeath=re.sub(r'(\d{3})(?=\d)', r'\1,', str(totaldeath)[::-1])[::-1]

    atleastonedose=int(totalvacc['people_vaccinated'].tail(1))
    atleastonedose=re.sub(r'(\d{3})(?=\d)', r'\1,', str(atleastonedose)[::-1])[::-1]
    totalfullyvacci=int(totalvacc['people_fully_vaccinated'].tail(1))
    totalfullyvacci=re.sub(r'(\d{3})(?=\d)', r'\1,', str(totalfullyvacci)[::-1])[::-1]
    totalvacci=int(totalvacc['total_vaccinations'].tail(1))
    totalvacci=re.sub(r'(\d{3})(?=\d)', r'\1,', str(totalvacci)[::-1])[::-1]

    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    c1, c2, c3= st.columns(3)
    c1.metric("Total Confirmed", totalconfirme)
    c2.metric("Total Death", totaldeath)
    c3.metric('Total doses administered',totalvacci)

    cc1, cc2= st.columns(2)
    cc1.metric("Total Fully Vaccinated", totalfullyvacci)
    cc2.metric("Total people Vaccinated at least one dose", atleastonedose)


    import geopandas as gpd



    oldcovid19_DataFram['Province/State']=oldcovid19_DataFram['Province/State'].fillna('')
    oldcovid19_DataFram['Country/Region']=oldcovid19_DataFram['Country/Region'].astype(str) +oldcovid19_DataFram['Province/State'].astype(str) 


    oldcovid19_DataFram = oldcovid19_DataFram.replace({'Country/Region' : 'Gambia'}, 'The Gambia')
    oldcovid19_DataFram = oldcovid19_DataFram.replace({'Country/Region' : 'DenmarkGreenland'}, 'Greenland')
    oldcovid19_DataFram = oldcovid19_DataFram.replace({'Country/Region' : 'Korea, South'}, 'Republic of Korea')
    oldcovid19_DataFram = oldcovid19_DataFram.replace({'Country/Region' : 'Russia'}, 'Russian Federation')
    oldcovid19_DataFram = oldcovid19_DataFram.replace({'Country/Region' : 'Czechia'}, 'Czech Republic')
    oldcovid19_DataFram = oldcovid19_DataFram.replace({'Country/Region' : 'Taiwan*'}, 'Taiwan')
    oldcovid19_DataFram = oldcovid19_DataFram.replace({'Country/Region' : 'North Macedonia'}, 'Macedonia')
    oldcovid19_DataFram = oldcovid19_DataFram.replace({'Country/Region' : 'Gambia'}, 'The Gambia')
    oldcovid19_DataFram = oldcovid19_DataFram.replace({'Country/Region' : 'Congo (Kinshasa)'}, 'Democratic Republic of the Congo')
    oldcovid19_DataFram = oldcovid19_DataFram.replace({'Country/Region' : 'Congo (Brazzaville)'}, 'Republic of Congo')
    oldcovid19_DataFram = oldcovid19_DataFram.replace({'Country/Region' : "Laos"}, "Lao PDR")
    oldcovid19_DataFram = oldcovid19_DataFram.replace({'Country/Region' : 'Burma'}, "Myanmar")
    oldcovid19_DataFram = oldcovid19_DataFram.replace({'Country/Region' : 'Brunei'}, "Brunei Darussalam")

    oldcovid19_DataFram = oldcovid19_DataFram.replace({'Country/Region' :['ChinaAnhui', 'ChinaBeijing', 'ChinaChongqing',
           'ChinaFujian', 'ChinaGansu', 'ChinaGuangdong', 'ChinaGuangxi',
           'ChinaGuizhou', 'ChinaHainan', 'ChinaHebei', 'ChinaHeilongjiang',
           'ChinaHenan', 'ChinaHong Kong', 'ChinaHubei', 'ChinaHunan',
           'ChinaInner Mongolia', 'ChinaJiangsu', 'ChinaJiangxi',
           'ChinaJilin', 'ChinaLiaoning', 'ChinaMacau', 'ChinaNingxia',
           'ChinaQinghai', 'ChinaShaanxi', 'ChinaShandong', 'ChinaShanghai',
           'ChinaShanxi', 'ChinaSichuan', 'ChinaTianjin', 'ChinaTibet',
           'ChinaUnknown', 'ChinaXinjiang', 'ChinaYunnan', 'ChinaZhejiang',]},'China')

    oldcovid19_DataFram = oldcovid19_DataFram.replace({'Country/Region' :['AustraliaAustralian Capital Territory',
           'AustraliaNew South Wales', 'AustraliaNorthern Territory',
           'AustraliaQueensland', 'AustraliaSouth Australia',
           'AustraliaTasmania', 'AustraliaVictoria',
           'AustraliaWestern Australia']},'Australia')

    oldcovid19_DataFram = oldcovid19_DataFram.replace({'Country/Region' :['FranceFrench Guiana', 'FranceFrench Polynesia',
           'FranceGuadeloupe', 'FranceMartinique', 'FranceMayotte',
           'FranceNew Caledonia', 'FranceReunion', 'FranceSaint Barthelemy',
           'FranceSaint Pierre and Miquelon', 'FranceSt Martin',
           'FranceWallis and Futuna', 'France']},'France')
    oldcovid19_DataFram = oldcovid19_DataFram.replace({'Country/Region' :['CanadaAlberta', 'CanadaBritish Columbia',
           'CanadaDiamond Princess', 'CanadaGrand Princess', 'CanadaManitoba',
           'CanadaNew Brunswick', 'CanadaNewfoundland and Labrador',
           'CanadaNorthwest Territories', 'CanadaNova Scotia',
           'CanadaNunavut', 'CanadaOntario', 'CanadaPrince Edward Island',
           'CanadaQuebec', 'CanadaRepatriated Travellers',
           'CanadaSaskatchewan', 'CanadaYukon']},'Canada')






    covidd=oldcovid19_DataFram.groupby('Country/Region').tail(1)
    covidd.reset_index(drop=True,inplace=True)
    covidd[['deaths percentage','confirmed percentage']]=covidd[['deaths percentage','confirmed percentage']].astype(str).add(' %')


    jsonfile=gpd.read_file('https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/public/data/_work-in-progress/world.geojson')

    def getconfirmed (df):
        a = covidd["Country/Region"]
        if df in list(a):
            i = list(a).index(df)
            return covidd.loc[i, "Confirmed"]
        else:
            return 0
    def getconfirmedperc (df):
        a = covidd["Country/Region"]
        if df in list(a):
            i = list(a).index(df)
            return covidd.loc[i, "confirmed percentage"]
        else:
            return 0
    def getdeaths (df):
        a = covidd["Country/Region"]
        if df in list(a):
            i = list(a).index(df)
            return covidd.loc[i, "deaths"]
        else:
            return 
    def getdeathsperc (df):
        a = covidd["Country/Region"]
        if df in list(a):
            i = list(a).index(df)
            return covidd.loc[i, "deaths percentage"]
        else:
            return 0
        


    jsonfile["Confirmed"] = jsonfile["name"].apply(getconfirmed)
    jsonfile["Deaths"] = jsonfile["name"].apply(getdeaths)
    jsonfile["confirmed percentage"] = jsonfile["name"].apply(getconfirmedperc)
    jsonfile["deaths percentage"] = jsonfile["name"].apply(getdeathsperc)

    w=[oldcovid19_DataFram['Lat'].values[0],oldcovid19_DataFram['Long'].values[0]]

    locat =folium.Map(w,zoom_start=1.6,min_zoom=2)
    folium.Choropleth(geo_data=jsonfile,
                      data = jsonfile,
                      columns=['name','Confirmed'],
                      color='Confirmed',
                      key_on="feature.properties.name",
                      fill_color='Set1',
                      highlight=True).add_to(locat)

    style_function = lambda x: {'fillColor': '#ffffff', 
                                'color':'#000000', 
                                'fillOpacity': 0.1, 
                                'weight': 0.1}

    highlight_function = lambda x: {'fillColor': '#000000', 
                                    'color':'#000000', 
                                    'fillOpacity': 0.50, 
                                    'weight': 0.1}
    folium.features.GeoJson(
        jsonfile, 
        style_function=style_function, 
        control=False,
        highlight_function=highlight_function, 
        tooltip=folium.features.GeoJsonTooltip(
            fields=['name','Confirmed',"confirmed percentage",'Deaths',"deaths percentage"],
            aliases=['Country: ','Confirmed: ',"confirmed percentage: ",'Deaths: ',"deaths percentage: "]
        )
    ).add_to(locat)

    folium_static(locat,width=1000,height=600)
elif my_model =='Oman Cases':
    st.markdown("<h1 style='text-align: center;'>Oman Case</h1>", unsafe_allow_html=True)

    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    covid19_DataFram[['New Cases','New Deaths']]=covid19_DataFram[['New Cases','New Deaths']].astype(int)

    Omancases=covid19_DataFram[covid19_DataFram['Country/Region']=='Oman']
    Omanvaccin=covid_vaccin[covid_vaccin['location']=='Oman']

    omanconf=Omancases['Confirmed'].tail(1).to_string(index=False)

    omanconfperc=Omancases['confirmed percentage'].tail(1).to_string(index=False)

    omandeat=Omancases['deaths'].tail(1).to_string(index=False)

    omandeatperc=Omancases['deaths percentage'].tail(1).to_string(index=False)

    totalvaccin=Omanvaccin['total_vaccinations'].tail(1).to_string(index=False)

    onedosevaccin=Omanvaccin['people_vaccinated'].tail(1).to_string(index=False)

    fullvaccin=Omanvaccin['people_fully_vaccinated'].tail(1).to_string(index=False)

    c1, c2, c3= st.columns(3)
    c1.metric("Total Confirmed", omanconf)
    c2.metric("Total Death", omandeat)
    c3.metric('Total doses administered',totalvaccin)

    c11, c21,c31,c41= st.columns(4)
    c11.metric("confirmed percentage",omanconfperc+"%")
    c21.metric("deaths percentage",omandeatperc+"%")
    c31.metric("Vaccinated at least one dose", onedosevaccin)
    c41.metric("Total Fully Vaccinated", fullvaccin)

    selectbox = st.selectbox("Select the option",('Day and Month', 'Day','Date and Month','Date'))

    ConfirmedDM= alt.Chart(Omancases,width=700,height=500).mark_bar().encode(
        x="day(Date):O",
        y="month(Date):O",
        color="sum(New Cases)",
        tooltip="sum(New Cases)"
    )

    ConfirmedDMtext=alt.Chart(Omancases,width=700,height=500).mark_text().encode(
        x="day(Date):O",
        y="month(Date):O",
        text="sum(New Cases)" 
    )

    ConfirmedDay= alt.Chart(Omancases,width=700,height=150).mark_bar().encode(
        x="day(Date):O",
       # y="month(Date):O",
        color="sum(New Cases)",
        tooltip="sum(New Cases)"
    )

    ConfirmedDaytext=alt.Chart(Omancases,width=700,height=150).mark_text().encode(
        x="day(Date):O",
        #y="month(Date):O",
        text="sum(New Cases)" 
    )


    ConfirmedDateM= alt.Chart(Omancases,width=900,height=500).mark_bar().encode(
        x="date(Date):O",
        y="month(Date):O",
        color="sum(New Cases)",
        tooltip="sum(New Cases)"
    )

    ConfirmedDateMtext=alt.Chart(Omancases,width=900,height=500).mark_text(angle=270).encode(
        x="date(Date):O",
        y="month(Date):O",
        text="sum(New Cases)" 
    )

    ConfirmedDate= alt.Chart(Omancases,width=900,height=100).mark_bar().encode(
        x="date(Date):O",
       # y="month(Date):O",
        color="sum(New Cases)",
        tooltip="sum(New Cases)"
    )

    ConfirmedDatetext=alt.Chart(Omancases,width=900,height=100).mark_text(angle=270).encode(
        x="date(Date):O",
        #y="month(Date):O",
        text="sum(New Cases)" 
    )

    deathsDM= alt.Chart(Omancases,width=700,height=500).mark_bar().encode(
        x="day(Date):O",
        y="month(Date):O",
        color="sum(New Deaths)",
        tooltip="sum(New Deaths)"
    )

    deathsDMtext=alt.Chart(Omancases,width=700,height=500).mark_text().encode(
        x="day(Date):O",
        y="month(Date):O",
        text="sum(New Deaths)" 
    )

    deathsDay= alt.Chart(Omancases,width=700,height=150).mark_bar().encode(
        x="day(Date):O",
       # y="month(Date):O",
        color="sum(New Deaths)",
        tooltip="sum(New Deaths)"
    )

    deathsDaytext=alt.Chart(Omancases,width=700,height=150).mark_text().encode(
        x="day(Date):O",
        #y="month(Date):O",
        text="sum(New Deaths)" 
    )


    deathsDateM= alt.Chart(Omancases,width=900,height=500).mark_bar().encode(
        x="date(Date):O",
        y="month(Date):O",
        color="sum(New Deaths)",
        tooltip="sum(New Deaths)"
    )

    deathsDateMtext=alt.Chart(Omancases,width=900,height=500).mark_text(angle=270).encode(
        x="date(Date):O",
        y="month(Date):O",
        text="sum(New Deaths)" 
    )

    deathsDate= alt.Chart(Omancases,width=900,height=100).mark_bar().encode(
        x="date(Date):O",
       # y="month(Date):O",
        color="sum(New Deaths)",
        tooltip="sum(New Deaths)"
    )

    deathsDatetext=alt.Chart(Omancases,width=900,height=100).mark_text(angle=270).encode(
        x="date(Date):O",
        #y="month(Date):O",
        text="sum(New Deaths)" 
    )




    if selectbox == 'Day and Month':
         st.write("## View Confirmed for Oman")
         st.altair_chart(ConfirmedDM+ConfirmedDMtext)
         st.write("## View deaths for Oman")
         st.altair_chart(deathsDM+deathsDMtext)
    elif selectbox == 'Day':
        st.write("## View Confirmed for Oman")
        st.altair_chart(ConfirmedDay+ConfirmedDaytext)
        st.write("## View deaths for Oman")
        st.altair_chart(deathsDay+deathsDaytext)
    elif selectbox == 'Date and Month':
        st.write("## View Confirmed for Oman")
        st.altair_chart(ConfirmedDateM+ConfirmedDateMtext)
        st.write("## View deaths for Oman")
        st.altair_chart(deathsDateM+deathsDateMtext)
    else:
       st.write("## View Confirmed for Oman")
       st.altair_chart(ConfirmedDate+ConfirmedDatetext)
       st.write("## View deaths for Oman")
       st.altair_chart(deathsDate+deathsDatetext)

    ques=st.selectbox("select your option:", ['Oman Map',"Dates with the most new cases",'Dates with the most new deaths',"Most month New Cases","Most month New Deaths","Information about Oman"])

    if ques =='Oman Map':
         
        import requests
        from bs4 import BeautifulSoup
        import re
        import folium
        import geopandas as gpd
        from streamlit_folium import folium_static
        webhtml= "https://tekany.net/%D8%B9%D9%8F%D9%85%D8%A7%D9%86%D9%8A%D8%A7%D8%AA/%D8%AA%D9%81%D8%A7%D8%B5%D9%8A%D9%84-%D8%A5%D8%B5%D8%A7%D8%A8%D8%A7%D8%AA-%D9%81%D9%8A%D8%B1%D9%88%D8%B3-%D9%83%D9%88%D8%B1%D9%88%D9%86%D8%A7-%D9%81%D9%8A-%D8%B3%D9%84%D8%B7%D9%86%D8%A9-%D8%B9%D9%85/.html"
        web_text = requests.get(webhtml).text
        soup = BeautifulSoup(web_text,'html')
        table= soup.find_all("table", attrs={"class": "tablepress tablepress-id-5"})
        table = table[0]
        bdy= table.find_all("tr")
        hd = bdy[0] 
        bdy_rows = bdy[1:] 
        Columnsnames= []
        for item in hd.find_all("th"):
            item = (item.text).rstrip("\n")
            Columnsnames.append(item)
        
        listallrows= []
        for row_num in range(len(bdy_rows)): 
            row = [] 
            for row_item in bdy_rows[row_num].find_all("td"): 
      
                a = re.sub("(\xa0)|(\n)|,","",row_item.text)
                row.append(a)
        
            listallrows.append(row)
        dfoman = pd.DataFrame(data=listallrows,columns=Columnsnames)
        dfoman.rename(columns ={'المحافظة':'Governorate',	'الإصابات الجديدة':'New_cases','إجمالي الحالات المصابة':'confirmed','الحالات التي تماثلت بالشفاء':'recovered','عدد حالات الوفاة':'Deaths','عدد الحالات النشطة':'active'}, inplace = True)
        dfoman= dfoman.replace({'Governorate' : 'مسقط'}, 'Muscat')
        dfoman= dfoman.replace({'Governorate' : 'شمال الباطنة'}, 'Al Batnah')
        dfoman= dfoman.replace({'Governorate' : 'جنوب الباطنة'}, 'Al Batnah')
        dfoman= dfoman.replace({'Governorate' : 'الداخلية'}, 'Ad Dakhliyah')
        dfoman= dfoman.replace({'Governorate' : 'ظفار'}, 'Dhofar')
        dfoman= dfoman.replace({'Governorate' : 'جنوب الشرقية'}, 'Ash Sharqiyah')
        dfoman= dfoman.replace({'Governorate' : 'الظاهرة'}, 'Al Dhahira')
        dfoman= dfoman.replace({'Governorate' : 'شمال الشرقية'}, 'Ash Sharqiyah')
        dfoman= dfoman.replace({'Governorate' : 'الوسطى'}, 'Al Wusta')
        dfoman= dfoman.replace({'Governorate' : 'البريمي'}, 'Albarimi')
        dfoman= dfoman.replace({'Governorate' : 'مسندم'}, 'Musandam')
        dfoman=dfoman.dropna()
        dfoman['New_cases'] = pd.to_numeric(dfoman['New_cases'])
        dfoman['confirmed'] = pd.to_numeric(dfoman['confirmed'])
        dfoman['recovered'] = pd.to_numeric(dfoman['recovered'])
        dfoman['Deaths'] = pd.to_numeric(dfoman['Deaths'])
        dfoman['active'] = pd.to_numeric(dfoman['active'])
        dfoman=dfoman.groupby('Governorate').sum()
        dfoman.reset_index(inplace=True)
        muscat = [23.614328,58.545284]
        jfile=gpd.read_file('https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/oman.geojson')
        

        def getnewcase (df):
            a = dfoman["Governorate"]
            if df in list(a):
                i = list(a).index(df)
                return dfoman.loc[i, "New_cases"]
            else:
                return 0
        def getconfirmed (df):
            a = dfoman["Governorate"]
            if df in list(a):
                i = list(a).index(df)
                return dfoman.loc[i, "confirmed"]
            else:
                return 0
        def getrecovered (df):
            a = dfoman["Governorate"]
            if df in list(a):
                i = list(a).index(df)
                return dfoman.loc[i, "recovered"]
            else:
                return 
        def getdeaths (df):
            a = dfoman["Governorate"]
            if df in list(a):
                i = list(a).index(df)
                return dfoman.loc[i, "Deaths"]
            else:
                return 0
        
        def getactive (df):
            a = dfoman["Governorate"]
            if df in list(a):
                i = list(a).index(df)
                return dfoman.loc[i, "active"]
            else:
                return 0
            


        jfile["New_cases"] = jfile["name"].apply(getnewcase)
        jfile["Deaths"] = jfile["name"].apply(getdeaths)
        jfile["confirmed"] = jfile["name"].apply(getconfirmed)
        jfile["recovered"] = jfile["name"].apply(getrecovered)
        jfile["active"] = jfile["name"].apply(getactive)
        


        mm=folium.Map(muscat,zoom_start=6,max_zoom=6,min_zoom=6)
        folium.Choropleth(geo_data=jfile,
                          data = jfile,
                          columns=['name','confirmed'],
                          color='confirmed',
                          key_on="feature.properties.name",
                          fill_color='Dark2',
                          highlight=True).add_to(mm)

        style_function = lambda x: {'fillColor': '#ffffff', 
                                    'color':'#000000', 
                                    'fillOpacity': 0.1, 
                                    'weight': 0.1}

        highlight_function = lambda x: {'fillColor': '#000000', 
                                        'color':'#000000', 
                                        'fillOpacity': 0.50, 
                                        'weight': 0.1}
        folium.features.GeoJson(
            jfile, 
            style_function=style_function, 
            control=False,
            highlight_function=highlight_function, 
            tooltip=folium.features.GeoJsonTooltip(
                fields=['name','confirmed','Deaths',"recovered",'New_cases',"active"],
            )
        ).add_to(mm)
        
        folium_static(mm,width=800,height=600)
        st.write("\n")
        st.write("\n")
        st.write(dfoman)
        
        
        
        
        
        
    elif ques == "Dates with the most new cases":
         st.write("The diagram below shows the dates that contain the most new cases")
         topOman=Omancases.groupby('New Cases').max().sort_values("New Cases", ascending=False)
         topOman.reset_index(inplace=True)
         topOman=topOman[['Date','New Cases']].head(10)
         topOman.index += 1
         topOman['Date'] = topOman['Date'].dt.date
         
         chartnewcase = alt.Chart(topOman,width=700,height=500).mark_bar(color='red').encode(
            x=alt.X("Date"),
            y="New Cases",
            tooltip = ['Date','New Cases']
           ).interactive()
         st.altair_chart(chartnewcase)
         st.write(topOman)
         
    elif ques == "Dates with the most new deaths":
         st.write("The diagram below shows the dates that contain the most new deaths")
         topdeathOman=Omancases.groupby('New Deaths').max().sort_values("New Deaths", ascending=False)[:10]
         topdeathOman.reset_index(inplace=True)
         topdeathOman=topdeathOman[['Date','New Deaths']].head(10)
         topdeathOman.index += 1
         topdeathOman['Date'] = topdeathOman['Date'].dt.date
         chardeath = alt.Chart(topdeathOman,width=700,height=500).mark_bar(color='green').encode(
            x=alt.X("Date"),
            y="New Deaths",
            tooltip = ['Date','New Deaths']
           ).interactive()
         st.altair_chart(chardeath)
         st.write(topdeathOman)
         
    elif ques == "Most month New Cases":
         month=Omancases.groupby(Omancases['Date'].dt.strftime('%B-%Y')).sum()
         month.reset_index(inplace=True)
         m=month.groupby('New Cases').max().sort_values("New Cases", ascending=False)
         m.reset_index(inplace=True)
         m=m[['Date','New Cases']]
         chartM=alt.Chart(m,width=700,height=500).mark_bar(color='pink').encode(
            x=alt.X("Date"),
            y="New Cases",
            tooltip = ['Date','New Cases']
           ).interactive()
         st.altair_chart(chartM)
         st.write(m)
    elif ques == "Most month New Deaths":
        month=Omancases.groupby(Omancases['Date'].dt.strftime('%B-%Y')).sum()
        month.reset_index(inplace=True)
        m=month.groupby('New Deaths').max().sort_values("New Deaths", ascending=False)
        m.reset_index(inplace=True)
        m=m[['Date','New Deaths']]
        chartM=alt.Chart(m,width=700,height=500).mark_bar(color='red').encode(
           x=alt.X("Date"),
           y="New Deaths",
           tooltip = ['Date','New Deaths']
          ).interactive()
        st.altair_chart(chartM)
        st.write(m)
    else:
         st.write('###  Types of vaccine that Oman deal with:')
         st.write('BioNTech and Pfizer vaccine')
         st.write('### Number of doses the person have to take in Oman are:')
         st.write('Untill now 2 doeses')
         st.write('### What are the places where the vaccine was given in Oman?')
         st.write('Sports complex, governors office, hospital, health center')
         st.write('### The protection measures taken by the Sultanate of Oman to reduce Covid 19 ')
         st.write('Banning gatherings and events, closing many stores and keeping food stores, setting a ban on roaming more than once and for long periods, setting a fine for those who do not wear a mask, distributing sterilizers in all institutions, making studies and some work online')
         st.write('### Ages that should take the vaccine in the Sultanate of Oman?')
         st.write(' The Corona pandemic caused a decline in tourism and development indicators and a deterioration of the economy')



elif my_model == 'Information about COVID-19 pandemic':
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

elif my_model == 'The Most':
    
   st.markdown("<h1 style='text-align: center;'>The Most Top 10 countries</h1>", unsafe_allow_html=True)
   top10 = st.selectbox("Select your option",["Confirmed Cases","Deaths","Recovered"])
   typeofchart=['Bar Chart','Rule Chart','Circle Chart']
   selectchart=st.radio('select type of chart',typeofchart)
   
   topconf  = covid19_DataFram.groupby('Country/Region').max().sort_values(by='Confirmed', ascending=False)[:10]
   topconf.reset_index(inplace=True)

    
   ch1conf = alt.Chart(topconf,width=800,height=600).encode(
      x=alt.X("Country/Region", sort='-y'),
      y="Confirmed",
      color="Country/Region",
      tooltip = ["Confirmed","Country/Region"]
     ).interactive()
   
   conf_circle=alt.Chart(topconf,width=800,height=600).mark_circle().encode(
      x=alt.X("Country/Region"),
      size=alt.Size("Confirmed",scale=alt.Scale(domain=[1000,1000000], range=[0,300]),legend=None),
      color="Country/Region",
      tooltip = ["Confirmed","Country/Region"]
     ).interactive()
   
   conf_text=alt.Chart(topconf,width=800,height=600).mark_text(angle=360).encode(
      x=alt.X("Country/Region"),
      text="Country/Region",
      tooltip = ["Confirmed","Country/Region"]
     ).interactive()
   
   topdea = covid19_DataFram.groupby('Country/Region').max().sort_values(by='deaths', ascending=False)[:10]
   topdea.reset_index(inplace=True)
   ch2dea = alt.Chart(topdea,width=800,height=600).encode(
    x=alt.X("Country/Region", sort='-y'),
    y="deaths",
    color="Country/Region",
    tooltip = ["deaths","Country/Region"]
    ).interactive()

   de_circle=alt.Chart(topdea,width=800,height=600).mark_circle().encode(
      x=alt.X("Country/Region"),
      size=alt.Size("deaths",scale=alt.Scale(domain=[1,100000], range=[0,600]),legend=None),
      color="Country/Region",
      tooltip = ["deaths","Country/Region"]
     ).interactive()
   de_text=alt.Chart(topdea,width=800,height=600).mark_text(angle=360).encode(
      x=alt.X("Country/Region"),
      text="Country/Region",
      tooltip = ["deaths","Country/Region"]
     ).interactive()

   toprecov  = recov.groupby('Country/Region').max().sort_values(by='recovered', ascending=False)[:10]
   toprecov.reset_index(inplace=True)
   
   ch3recov = alt.Chart(toprecov,width=800,height=600).encode(
      x=alt.X("Country/Region", sort='-y'),
      y="recovered",
      color="Country/Region",
      tooltip = ["recovered","Country/Region"]
     ).interactive()
   recovered_circle = alt.Chart(toprecov,width=800,height=600).mark_circle().encode(
      x=alt.X("Country/Region"),
      size=alt.Size("recovered",scale=alt.Scale(domain=[1000,1000000], range=[0,300]),legend=None),
      color="Country/Region",
      tooltip = ["recovered","Country/Region"]
     ).interactive()
   recovered_text = alt.Chart(toprecov,width=800,height=600).mark_text(angle=360).encode(
      x=alt.X("Country/Region"),
      text="Country/Region",
      tooltip = ["recovered","Country/Region"]
     ).interactive()


   
   if top10 == "Confirmed Cases":
    if selectchart == 'Bar Chart':
        st.altair_chart(ch1conf.mark_bar())
        
    elif selectchart == 'Rule Chart':
        st.altair_chart(ch1conf.mark_rule())
       
    else:
        st.altair_chart(conf_circle+conf_text)
        
    
   elif top10 == "Deaths":
     if selectchart == 'Bar Chart':
        st.altair_chart(ch2dea.mark_bar())
     elif selectchart == 'Rule Chart':
        st.altair_chart(ch2dea.mark_rule())
     else:
        st.altair_chart(de_circle+de_text)
        
        
   else:
     if selectchart == 'Bar Chart':
        st.altair_chart(ch3recov.mark_bar())
     elif selectchart == 'Rule Chart':
        st.altair_chart(ch3recov.mark_rule())
     else:
        st.altair_chart(recovered_circle+recovered_text)
        
       
   st.write('# ------------------------------------------------')
   
    
   slider=st.slider("Pick Number to Display Top and Lowest Countries:",min_value=1,max_value=20,value=3)
   
   da=covid19_DataFram['Date'].tail(1).to_string(index=False)
   
   
   dfmin=covid19_DataFram.groupby('Country/Region').tail(1)
   dfmin.reset_index(inplace=True)
   
   if top10=='Confirmed Cases':
      maxconf= covid19_DataFram.groupby('Country/Region').max().sort_values(by='Confirmed', ascending=False)[:slider]
      maxconf.reset_index(inplace=True)
      maxconf.index += 1
      
      lowestconf=dfmin.sort_values('Confirmed',ascending=True)[:slider]
      lowestconf.reset_index(inplace=True)
      lowestconf.index += 1
      
      maxtable=maxconf[['Country/Region','Confirmed']]
      maxchart = alt.Chart(maxtable,width=800,height=600).mark_bar().encode(
          x=alt.X("Country/Region", sort='-y'),
          y=alt.Y("Confirmed",axis=alt.Axis(grid=False)),
          color="Country/Region",
          tooltip = ["Confirmed","Country/Region"]
          ).interactive()
      lowesttable=lowestconf[['Country/Region','Confirmed']]
      minchart = alt.Chart(lowesttable,width=800,height=600).mark_bar().encode(
          x=alt.X("Country/Region", sort='-y'),
          y=alt.Y("Confirmed",axis=alt.Axis(grid=False)),
          color="Country/Region",
          tooltip = ["Confirmed","Country/Region"]
          ).interactive()
   elif top10 == "Deaths":
       
     maxdea = covid19_DataFram.groupby('Country/Region').max().sort_values(by='deaths', ascending=False)[:slider]
     maxdea.reset_index(inplace=True)
     maxdea.index += 1
      
     lowestdea=dfmin.sort_values('deaths',ascending=True)[:slider]
     lowestdea.reset_index(inplace=True)
     lowestdea.index += 1
     
     maxtable=maxdea[['Country/Region','deaths']]
     
     maxchart = alt.Chart(maxtable,width=800,height=600).mark_bar().encode(
         x=alt.X("Country/Region", sort='-y'),
         y=alt.Y("deaths",axis=alt.Axis(grid=False)),
         color="Country/Region",
         tooltip = ["deaths","Country/Region"]
         ).interactive()
     
     lowesttable=lowestdea[['Country/Region','deaths']]
     
     minchart = alt.Chart(lowesttable,width=800,height=600).mark_bar().encode(
         x=alt.X("Country/Region", sort='-y'),
         y=alt.Y("deaths",axis=alt.Axis(grid=False)),
         color="Country/Region",
         tooltip = ["deaths","Country/Region"]
         ).interactive()
   else:
    toprecovtab= recov.groupby('Country/Region').max().sort_values(by='recovered', ascending=False)[:slider]
    toprecovtab.reset_index(inplace=True)
    toprecovtab.index += 1
    
    recovmin=recov.groupby('Country/Region').tail(1)
    recovmin.reset_index(inplace=True)
    recovmin.index += 1
    lowestrecov=recovmin.sort_values('recovered',ascending=True)[:slider]
    
    maxtable=toprecovtab[['Country/Region','recovered']]
    
    maxchart = alt.Chart(maxtable,width=800,height=600).mark_bar().encode(
      x=alt.X("Country/Region", sort='-y'),
      y=alt.Y("recovered",axis=alt.Axis(grid=False)),
      color="Country/Region",
      tooltip = ["recovered","Country/Region"]
     ).interactive()
    lowesttable=lowestrecov[['Country/Region','recovered']]
    minchart = alt.Chart(lowesttable,width=800,height=600).mark_bar().encode(
      x=alt.X("Country/Region", sort='-y'),
      y=alt.Y("recovered",axis=alt.Axis(grid=False)),
      color="Country/Region",
      tooltip = ["recovered","Country/Region"]
     ).interactive()

    
   st.write(f"### Top {slider} countries for {top10} in covid 19 from 2020-01-22 to	{da}." )
   st.altair_chart(maxchart)
   st.table(maxtable)
   st.write(f"### Lowest {slider} countries for {top10} in covid 19 from 2020-01-22	to {da}." )
   st.altair_chart(minchart)
   st.table(lowesttable)

elif my_model == 'Country wise analysis':
    
    st.markdown("<h1 style='text-align: center;'>Select the Country to display Chart</h1>", unsafe_allow_html=True)
    select_country = st.selectbox("Select country: ",covid19_DataFram["Country/Region"].unique())
    option = ['Confirmed','Deaths','Recovered']
    select_option =st.radio('Select your option', option)
    selchart =st.selectbox('Select type of Chart', ['Line Chart','Area Chart'])
    sc=covid19_DataFram[covid19_DataFram["Country/Region"]== select_country]
    infication= alt.Chart(sc,title="Confirmed cases",width=600,height=400).encode(
    x="yearmonth(Date)",
    y=alt.Y("Confirmed",title="Confirmed cases"),
    tooltip=['Confirmed',"yearmonth(Date)"]
    ).interactive()
    
    Confirmedprec= alt.Chart(sc,title="confirmed percentage",width=600,height=400).encode(
    x="yearmonth(Date)",
    y=alt.Y("confirmed percentage",title="confirmed percentage"),
    tooltip=['confirmed percentage',"yearmonth(Date)"]
    ).interactive()
    
    death= alt.Chart(sc,title="deaths",width=600,height=400).encode(
     x="yearmonth(Date)",
     y=alt.Y("deaths",title="Deaths cases"),
     tooltip=['deaths',"yearmonth(Date)"]
    ).interactive()
   
    screcov=recov[recov["Country/Region"]== select_country]
    Recovered= alt.Chart(screcov,width=600,height=400).encode(
    x="yearmonth(Date)",
    y=alt.Y("recovered",title="Recovered cases"),
    tooltip=['recovered',"yearmonth(Date)"]
    ).interactive()
    
    
    sc[['confirmed percentage','deaths percentage']]=sc[['confirmed percentage','deaths percentage']].round(2).astype(str).add(" %")
    
   
  
    if select_option == 'Confirmed':
      anali=sc['Confirmed'].tail(1).to_string(index=False)
      day=sc['Date'].tail(1).to_string(index=False)
      p=sc['confirmed percentage'].tail(1).to_string(index=False)
      cp=sc[['confirmed percentage','Country/Region']].tail(1)
      confper= alt.Chart(cp,title="confirmed percentage",width=600,height=400).mark_circle(color="teal",size=10000).encode(
       x="Country/Region",
       tooltip=['confirmed percentage',"Country/Region"]
      ).interactive()
      
      textper=alt.Chart(cp,width=600,height=400).mark_text(size=30,angle=360).encode(
         x=alt.X("Country/Region"),
         text="confirmed percentage",
         tooltip = ["confirmed percentage","Country/Region"]
        ).interactive()
      
      if selchart == 'Line Chart':
        st.altair_chart(infication.mark_line(color='green')|confper+textper)
      else:
        st.altair_chart(infication.mark_area(color='red')|confper+textper)
        
      st.write(f"The {selchart} above shows that the number of {select_option} in {select_country} is {anali} and with percentage of {p} until day {day}")
        

    elif select_option == 'Deaths':
      anali=sc['deaths'].tail(1).to_string(index=False)
      day=sc['Date'].tail(1).to_string(index=False)  
      cp=sc[['deaths percentage','Country/Region']].tail(1)
      p=sc['deaths percentage'].tail(1).to_string(index=False)
      deathper= alt.Chart(cp,title="deaths percentage",width=600,height=400).mark_circle(color="teal",size=10000).encode(
       x="Country/Region",
       tooltip=['deaths percentage',"Country/Region"]
      ).interactive()
      textper=alt.Chart(cp,width=600,height=400).mark_text(size=30,angle=360).encode(
         x=alt.X("Country/Region"),
         
         text="deaths percentage",
         tooltip = ["deaths percentage","Country/Region"]
        ).interactive()
      if selchart == 'Line Chart':
         st.altair_chart(death.mark_line(color='green')|deathper+textper)
      else:
        st.altair_chart(death.mark_area(color='red')|deathper+textper)
        
      st.write(f"The {selchart} above shows that number of {select_option} in {select_country} is {anali} and with a percentage of {p} until day {day}")
    

    else:
        anali=screcov['recovered'].tail(1).to_string(index=False)
        day=screcov['Date'].tail(1).to_string(index=False)
        if selchart == 'Line Chart':
          st.altair_chart(Recovered.mark_line(color='green'))
          
        else:
          st.altair_chart(Recovered.mark_area(color='red'))
        st.write(f"The {selchart} above shows that number of {select_option} in {select_country} are {anali} until day {day}")
    
    st.write("_________________________________________________________________________")      
    st.write("🟦 Total Confirmed , 🟥 Total Deaths ")
    st.altair_chart(infication.mark_bar()+death.mark_bar(color='red'))
    
elif my_model == 'Cumulative Data Visuals':

    st.markdown("<h1 style='text-align: center;'>Cumulative infection cases /deaths/recovered through months of the year</h1>", unsafe_allow_html=True)


    times=['Cumulative Confirmed cases','Cumulative deaths','Cumulative recovered']

    timeseries=st.selectbox('Select your option', times)

    typeofchart=['Line Chart','Area Chart']
    selectchart=st.radio('select type of chart',typeofchart)

    groupbydate=covid19_DataFram.groupby('Date').sum()
    groupbydate.reset_index(inplace=True)
    
    groupbydaterecov=recov.groupby('Date').sum()
    groupbydaterecov.reset_index(inplace=True)
    

    infectionCases=alt.Chart(groupbydate,width=700,height=600).encode(
    x=alt.X("Date",axis=alt.Axis(grid=False)),
    y=alt.Y("Confirmed",title="Cumulative infection cases",axis=alt.Axis(grid=False)),
    tooltip=['Confirmed','Date']
    ).interactive()

    death=alt.Chart(groupbydate,width=700,height=600).encode(
    x=alt.X("Date",axis=alt.Axis(grid=False)),
    y=alt.Y("deaths",title="Cumulative deaths",axis=alt.Axis(grid=False)),
    tooltip=['deaths',"Date"]
    ).interactive()

    recovered=alt.Chart(groupbydaterecov,width=700,height=600).encode(
    x=alt.X("Date",axis=alt.Axis(grid=False)),
    y=alt.Y("recovered",title="Cumulative recovered"),
    tooltip=['recovered',"Date"]
    ).interactive()

    if timeseries == 'Cumulative Confirmed cases':
      if selectchart == 'Line Chart':
        st.altair_chart(infectionCases.mark_line())
        cumulativevalue=groupbydate['Confirmed'].tail(1).to_string(index=False)
        lastday=groupbydate['Date'].tail(1).to_string(index=False)
      else:
        st.altair_chart(infectionCases.mark_area(color='red'))
        cumulativevalue=groupbydate['Confirmed'].tail(1).to_string(index=False)
        lastday=groupbydate['Date'].tail(1).to_string(index=False)

    elif timeseries == 'Cumulative deaths':
       if selectchart == 'Line Chart':
         st.altair_chart(death.mark_line())
         cumulativevalue=groupbydate['deaths'].tail(1).to_string(index=False)
         lastday=groupbydate['Date'].tail(1).to_string(index=False)
       else:
        st.altair_chart(death.mark_area(color='red'))
        cumulativevalue=groupbydate['deaths'].tail(1).to_string(index=False)
        lastday=groupbydate['Date'].tail(1).to_string(index=False)

    else:
      if selectchart == 'Line Chart':
        st.altair_chart(recovered.mark_line())
        cumulativevalue=groupbydaterecov['recovered'].tail(1).to_string(index=False)
        lastday=groupbydaterecov['Date'].tail(1).to_string(index=False)
      else:
        st.altair_chart(recovered.mark_area(color='red'))
        cumulativevalue=groupbydaterecov['recovered'].tail(1).to_string(index=False)
        lastday=groupbydaterecov['Date'].tail(1).to_string(index=False)

    st.write(f"The above {selectchart} shows that {timeseries} are {cumulativevalue} until date {lastday}")
    st.write("_____________________________________________________________________")
    st.write("🟩  Total Confirmed , 🟫 Total Deaths")
    if selectchart == 'Line Chart':
        st.altair_chart(infectionCases.mark_line(color="green")+death.mark_line(color="brown"))
    else:
        st.altair_chart(infectionCases.mark_area(color="green")+death.mark_area(color="brown"))
elif my_model == 'Compare Countries':
    
    st.markdown("<h1 style='text-align: center;'>Comparing Confirmed cases/ Deaths/ Recovered between multiple countries</h1>", unsafe_allow_html=True)


    multselect= st.multiselect('Select Multiple Countries',covid19_DataFram["Country/Region"].unique())
    cooption = ['Confirmed','Deaths','Recovered']
    slecttype=st.radio("Select Type :", cooption)
    cooptions=["scatter chart","circle chart"]
    selectchart=st.selectbox("Select Chart", cooptions)

    countryselected=covid19_DataFram[covid19_DataFram["Country/Region"].isin(multselect)]
    #for count country
    countcountry= countryselected.groupby('Country/Region').sum()
    countcountry.reset_index(inplace=True)
    cou=countcountry['Country/Region'].count()
    
    inf= alt.Chart(countryselected,width=800,height=600).encode(
        x="yearmonth(Date)",
        y="Country/Region",
        color="Country/Region",
        size='Confirmed',
        tooltip=["Country/Region",'Confirmed']
    
           ).interactive()

    dea= alt.Chart(countryselected,width=800,height=600).encode(
        x="yearmonth(Date)",
        y="Country/Region",
        color="Country/Region",
        size='deaths',
        tooltip=["Country/Region",'deaths']
    
        ).interactive()
    counselct=recov[recov["Country/Region"].isin(multselect)]
    rec= alt.Chart(counselct,width=800,height=600).encode(
        x="yearmonth(Date)",
        y="Country/Region",
        color="Country/Region",
        size='recovered',
        tooltip=["Country/Region",'recovered']
        ).interactive()

    circleConfirmed=alt.Chart(countryselected,width=800,height=600).mark_circle().encode(
        x="Country/Region",
        color="Country/Region",
        size=alt.Size('Confirmed',scale=alt.Scale(domain=[1,1000000], range=[1,1000]),legend=None),
        tooltip=["Country/Region",'Confirmed']
        ).interactive()

    circledea= alt.Chart(countryselected,width=800,height=600).mark_circle().encode(
        x="Country/Region",
        color="Country/Region",
        size=alt.Size('deaths',
                  scale=alt.Scale(domain=[1,10000],range=[1,1000]),legend=None),
        tooltip=["Country/Region",'deaths']
        ).interactive()

    circlerec= alt.Chart(counselct,width=800,height=600).mark_circle().encode(
        x="Country/Region",
        color="Country/Region",
        size=alt.Size('recovered',
                  scale=alt.Scale(domain=[1,100000], range=[1,1000]),legend=None),
        tooltip=["Country/Region",'recovered']
        ).interactive()

    newca=alt.Chart(countryselected,width=700,height=600).mark_line().encode(
        x=alt.X("Date",axis=alt.Axis(grid=False)),
        y=alt.Y("New Cases",axis=alt.Axis(grid=False)),
        color="Country/Region",
        tooltip=["Country/Region",'New Cases','Date']
        ).interactive()
    

    newdeath=alt.Chart(countryselected,width=700,height=600).mark_line().encode(
        x=alt.X("Date",axis=alt.Axis(grid=False)),
        y=alt.Y("New Deaths",axis=alt.Axis(grid=False)),
        color="Country/Region",
        tooltip=["Country/Region",'New Cases','Date']
        ).interactive()

    
    newrecovered=alt.Chart(counselct,width=700,height=600).mark_line().encode(
        x=alt.X("Date",axis=alt.Axis(grid=False)),
        y=alt.Y("New recovered",axis=alt.Axis(grid=False)),
        color="Country/Region",
        tooltip=["Country/Region",'New recovered','Date']
        ).interactive()
    
    pp=population[population['Country/Region'].isin(multselect)]
    pupchart=alt.Chart(pp,height=600).mark_bar().encode(
        x=alt.X("Country/Region",axis=alt.Axis(grid=False)),
        y=alt.Y("population",axis=alt.Axis(grid=False)),
        color="Country/Region",
        tooltip=["Country/Region",'population']
        ).interactive()
    
   
    
    if slecttype == 'Confirmed':
        #display country with value in table
        counana=countryselected[["Country/Region",'Confirmed']].tail(cou)
        counana.reset_index(inplace=True)
        counana.drop('index',axis=1,inplace=True)
        counana.sort_values(by='Confirmed', ascending=False,inplace=True)
        counana.reset_index(drop=True,inplace=True)
        counana.index += 1
        
        
        #higest and lowest 
        highest=counana[counana['Confirmed']== counana['Confirmed'].max()]
        higcountry= highest['Country/Region'].to_string(index=False)
        higvalue= highest['Confirmed'].to_string(index=False)
   
        lowest=counana[counana['Confirmed']== counana['Confirmed'].min()]
        lowcountry=lowest['Country/Region'].to_string(index=False)
        lowvalue=lowest['Confirmed'].to_string(index=False)
        #rank
        rankgroupbycountry=covid19_DataFram.groupby('Country/Region').tail(1)
        rankconf=rankgroupbycountry.sort_values('Confirmed',ascending=False)
        rankconf.reset_index(drop=True,inplace=True)
        rankconf.index += 1
        rankconf.reset_index(inplace=True)
        rankconf.rename(columns={'index': 'Confirmed Rank'},inplace=True)
        ra=rankconf[rankconf['Country/Region'].isin(multselect)]
        rank=ra[['Country/Region','Confirmed Rank']]
        rank.reset_index(drop=True,inplace=True)
        rank.index += 1
         
        if selectchart == 'scatter chart':
            st.altair_chart(inf.mark_circle())
            
        else:
            st.altair_chart(circleConfirmed)
            
        st.table(counana)
        st.write(f'The highest country chosen for Infection is **{higcountry}**, with a rate of **{higvalue}** and lowest country chosen for Infection is **{lowcountry}**, with a rate of **{lowvalue}**')
        st.write("## **The World Rank**")
        st.table(rank)
        st.altair_chart(newca|pupchart)
        
        
        fig = px.pie(countryselected, values=countryselected['confirmed percentage'].tail(cou), names=countryselected['Country/Region'].tail(cou), title='confirmed percentage')
        fig1 = px.pie(pp, values=pp["population"], names=pp['Country/Region'], title='population')
        
        st.plotly_chart(fig)
        st.plotly_chart(fig1)

        

    elif slecttype == 'Deaths':
        counana=countryselected[["Country/Region",'deaths']].tail(cou)
        counana.reset_index(inplace=True)
        counana.drop('index',axis=1,inplace=True)
        counana.sort_values(by='deaths', ascending=False,inplace=True)
        counana.reset_index(drop=True,inplace=True)
        counana.index += 1
   
        highest=counana[counana['deaths']== counana['deaths'].max()]
        higcountry= highest['Country/Region'].to_string(index=False)
        higvalue= highest['deaths'].to_string(index=False)
        lowest=counana[counana['deaths']== counana['deaths'].min()]
        lowcountry=lowest['Country/Region'].to_string(index=False)
        lowvalue=lowest['deaths'].to_string(index=False)
   
        #rank
        rankgroupbycountry=covid19_DataFram.groupby('Country/Region').tail(1)
        rankdea=rankgroupbycountry.sort_values('deaths',ascending=False)
        rankdea.reset_index(drop=True,inplace=True)
        rankdea.index += 1
        rankdea.reset_index(inplace=True)
        rankdea.rename(columns={'index': 'deaths Rank'},inplace=True)
        ra=rankdea[rankdea['Country/Region'].isin(multselect)]
        rank=ra[['Country/Region','deaths Rank']]
        rank.reset_index(drop=True,inplace=True)
        rank.index += 1
        
        
        if selectchart == 'scatter chart':
            st.altair_chart(dea.mark_circle())
            
        else:
            st.altair_chart(circledea|pupchart)
        st.table(counana)
        st.write(f'The highest country chosen for deaths is **{higcountry}**, with a rate of **{higvalue}** and lowest country chosen for deaths is **{lowcountry}**, with a rate of **{lowvalue}**')
        st.write("## **The World Rank**")
        st.table(rank)
        st.altair_chart(newdeath|pupchart)
        fig = px.pie(countryselected, values=countryselected["deaths percentage"].tail(cou), names=countryselected['Country/Region'].tail(cou), title='deaths percentage')
        st.plotly_chart(fig)
        fig1 = px.pie(pp, values=pp["population"], names=pp['Country/Region'], title='population')
        st.plotly_chart(fig1)
    else:
       counana=counselct[["Country/Region",'recovered']].tail(cou)
       counana.reset_index(inplace=True)
       counana.drop('index',axis=1,inplace=True)
       counana.sort_values(by='recovered', ascending=False,inplace=True)
       counana.reset_index(drop=True,inplace=True)
       counana.index += 1
       
       highest=counana[counana['recovered']== counana['recovered'].max()]
       higcountry= highest['Country/Region'].to_string(index=False)
       higvalue= highest['recovered'].to_string(index=False)
       lowest=counana[counana['recovered']== counana['recovered'].min()]
       lowcountry=lowest['Country/Region'].to_string(index=False)
       lowvalue=lowest['recovered'].to_string(index=False)
   
       rankgroupbycountry=recov.groupby('Country/Region').tail(1)
       rankrecov=rankgroupbycountry.sort_values('recovered',ascending=False)
       rankrecov.reset_index(drop=True,inplace=True)
       rankrecov.index += 1
       rankrecov.reset_index(inplace=True)
       rankrecov.rename(columns={'index': 'recovered Rank'},inplace=True)
       ra=rankrecov[rankrecov['Country/Region'].isin(multselect)]
       rank=ra[['Country/Region','recovered Rank']]
       rank.reset_index(drop=True,inplace=True)
       rank.index += 1
       
       if selectchart == 'scatter chart':
           st.altair_chart(rec.mark_circle())
          
         
       else:
           st.altair_chart(circlerec)
       st.table(counana)
       st.write(f'The highest country chosen for recovered is **{higcountry}**, with a rate of **{higvalue}** and lowest country chosen for recovered is **{lowcountry}**, with a rate of **{lowvalue}**')
       st.write("## **The World Rank**")
       st.table(rank)
       st.altair_chart(newrecovered|pupchart)
     
elif my_model == 'Vaccination':
    

    #Total world vaccinations
 
    st.markdown("<h1 style='text-align: center;'>Vaccination</h1>", unsafe_allow_html=True)
    st.write("### Total world vaccinations") 
    selectoptions=st.selectbox('select your option: ',['Total doses administered','people vaccinated (at least one dose)','people fully vaccinated','total boosters'])

    cumlativevacc=covid_vaccin[covid_vaccin['location']=='World']

    cumtadoese=alt.Chart(cumlativevacc,width=700,height=400).mark_line(color='navy').encode(
        x=alt.X("date",title='Date',axis=alt.Axis(grid=False)),
        y=alt.Y("total_vaccinations",title='Total doses administered',axis=alt.Axis(grid=False)),
        tooltip=['total_vaccinations','date']
        ).interactive()

    cumtfirstdose=alt.Chart(cumlativevacc,width=700,height=400).mark_line(color='pink').encode(
        x=alt.X("date",title='Date',axis=alt.Axis(grid=False)),
        y=alt.Y("people_vaccinated",title='people vaccinated (at least one dose)',axis=alt.Axis(grid=False)),
        tooltip=['people_vaccinated','date']
        ).interactive()
    
    cumtfullyvacc=alt.Chart(cumlativevacc,width=700,height=400).mark_line(color='aqua').encode(
        x=alt.X("date",title='Date',axis=alt.Axis(grid=False)),
        y=alt.Y("people_fully_vaccinated",title='people fully vaccinated',axis=alt.Axis(grid=False)),
        tooltip=['people_fully_vaccinated','date']
        ).interactive()
    
    cumttotalboster=alt.Chart(cumlativevacc,width=700,height=400).mark_line(color='brown').encode(
        x=alt.X("date",title='Date',axis=alt.Axis(grid=False)),
        y=alt.Y("total_boosters",title='total boosters',axis=alt.Axis(grid=False)),
        tooltip=['total_boosters','date']
        ).interactive()
    
    cumanali=cumlativevacc[['date','total_vaccinations','people_vaccinated','people_fully_vaccinated','total_boosters','daily_vaccinations']].tail(1)
    cumday=cumanali['date'].to_string(index=False)
    
    
    if selectoptions== 'Total doses administered':
        st.altair_chart(cumtadoese)
        doses=cumanali['total_vaccinations']
        doses=int(doses)
        st.write(f'Total doses administered until date {cumday}  are {doses}')
    elif selectoptions == 'people vaccinated (at least one dose)':
        st.altair_chart(cumtfirstdose)
        doses=cumanali['people_vaccinated']
        doses=int(doses)
        st.write(f'People Vaccinated (first dose) until date {cumday}  are {doses}')
    elif selectoptions =='people fully vaccinated':
        st.altair_chart(cumtfullyvacc)
        doses=cumanali['people_fully_vaccinated']
        doses=int(doses)
        st.write(f'People Fully Vaccinated until date {cumday}  are {doses}')
    else:
        st.altair_chart(cumttotalboster)
        doses=cumanali['total_boosters']
        doses=int(doses)
        st.write(f'Total Boosters until date {cumday}  are {doses}')
        
    #Vaccination by Select countries
    
    st.write("### Cumulative Vaccination by Select Countries")
    covidvacc=covid_vaccin[~(covid_vaccin['location']=='World')]
    selectcoun = st.selectbox("Select country: ",covidvacc["location"].unique())
    
    countryselected=covidvacc[covidvacc["location"]== selectcoun]
    groupdate=countryselected.groupby('date').sum()
    groupdate.reset_index(inplace=True)
    
    total_vacc=alt.Chart(groupdate,width=700,height=400).encode(
        x=alt.X("date",sort='-y',title='Date',axis=alt.Axis(grid=False)),
        y=alt.Y("total_vaccinations",axis=alt.Axis(grid=False)),
        tooltip=['total_vaccinations','date']
        ).interactive()
    
    people= alt.Chart(groupdate,width=700,height=400).encode(
        x=alt.X("date",sort='-y',title='Date',axis=alt.Axis(grid=False)),
        y=alt.Y("people_vaccinated",axis=alt.Axis(grid=False)),
        tooltip=['people_vaccinated','date']
        ).interactive()
    completed=alt.Chart(groupdate,width=700,height=400).encode(
        x=alt.X("date",sort='-y',title='Date',axis=alt.Axis(grid=False)),
        y=alt.Y("people_fully_vaccinated",axis=alt.Axis(grid=False)),
        tooltip=['people_fully_vaccinated','date']
        ).interactive()
    bosters=alt.Chart(groupdate,width=700,height=400).encode(
        x=alt.X("date",sort='-y',title='Date',axis=alt.Axis(grid=False)),
        y=alt.Y("total_boosters",axis=alt.Axis(grid=False)),
        tooltip=['total_boosters','date']
        ).interactive()
    
    lastvalue=groupdate.tail(1)
    perc=population[population['Country/Region'] == selectcoun]
    
    if selectoptions == 'Total doses administered':
         st.altair_chart(total_vacc.mark_area(color='turquoise'))
         value=lastvalue['total_vaccinations']
         value=int(value)
         st.write(f'Total doses administered in {selectcoun} are {value}')
    elif selectoptions == 'people vaccinated (at least one dose)':
         st.altair_chart(people.mark_area(color='maroon'))
         value=lastvalue['people_vaccinated']
         value=int(value)
         pop=perc['population']
         pop=int(pop)
         percent=(value/pop)*100
         percent=int(percent)
         st.write(f'People Vaccinated (at least one dose) in {selectcoun}  are {value} and with percent of {percent}%')
    elif selectoptions == 'people fully vaccinated':
         st.altair_chart(completed.mark_area(color='crimson'))
         value=lastvalue['people_fully_vaccinated']
         value=int(value)
         pop=perc['population']
         pop=int(pop)
         percent=(value/pop)*100
         percent=int(percent)
         st.write(f'People Fully Vaccinated in {selectcoun}  are {value} and with percent of {percent}%')
    else:
        st.altair_chart(bosters.mark_area(color='orchid'))
        value=lastvalue['total_boosters'].to_string(index=False)
        st.write(f'Total Boosters in {selectcoun}  are {value}')
    #daily_vaccinations
    st.write(f'### Daily vaccinations in {selectcoun}')  
    graph=st.selectbox('select chart', ['Line Chart',"Area Chart"])
    dailyvacc=alt.Chart(countryselected,width=700,height=400).encode(
        x=alt.X("date",title='Date',axis=alt.Axis(grid=False)),
        y=alt.Y("daily_vaccinations",axis=alt.Axis(grid=False)),
        tooltip=['daily_vaccinations','location','date']
        ).interactive()  
    if graph=='Line Chart':
        st.altair_chart(dailyvacc.mark_line(color='teal'))
    else:
        st.altair_chart(dailyvacc.mark_area(color='lime'))

    

    #Cumulative Vaccination vs Cumulative of cases

    st.write("### Cumulative and daily Vaccination vs  Cumulative and daily cases in the world")
    selectcase = st.selectbox("Select your option",["Confirmed Cases","Deaths","Recovered"])
    
    fully=alt.Chart(cumlativevacc,width=700,height=400).mark_line(color='green').encode(
        x=alt.X("date",axis=alt.Axis(grid=False)),
        y=alt.Y("people_fully_vaccinated",title='fully vaccinated',axis=alt.Axis(grid=False)),
        tooltip=['people_fully_vaccinated','date']
        ).interactive()
    
    vacc_daily=alt.Chart(cumlativevacc,width=700,height=400).mark_line(color='brown').encode(
        x=alt.X("date",axis=alt.Axis(grid=False)),
        y=alt.Y("daily_vaccinations",title='daily vaccinations',axis=alt.Axis(grid=False)),
        tooltip=['daily_vaccinations','date']
        ).interactive()
    
    
    groupbydate=covid19_DataFram.groupby('Date').sum()
    groupbydate.reset_index(inplace=True)
        
    groupbydaterecov=recov.groupby('Date').sum()
    groupbydaterecov.reset_index(inplace=True)
    
    cofvsvacc=alt.Chart(groupbydate,width=700,height=400).mark_line(color='red').encode(
        x="Date",
        y=alt.Y("Confirmed",title='Confirmed'),
        tooltip=['Confirmed','Date']
        ).interactive()
        
    dailycases=alt.Chart(groupbydate,width=700,height=400).mark_line().encode(
        x="Date",
        y=alt.Y('New Cases',title='New Cases'),
        tooltip=['New Cases','Date']
        ).interactive()
    
    deathvsvacc=alt.Chart(groupbydate,width=700,height=400).mark_line(color='red').encode(
        x="Date",
        y=alt.Y("deaths",title='deaths'),
        tooltip=['deaths',"Date"]
        ).interactive()
    
    dailydeath=alt.Chart(groupbydate,width=700,height=400).mark_line().encode(
        x="Date",
        y=alt.Y("New Deaths",title='New Deaths'),
        tooltip=['New Deaths','Date']
        ).interactive()
    recovvsvacc=alt.Chart(groupbydaterecov,width=700,height=400).mark_line(color='red').encode(
        x="Date",
        y=alt.Y("recovered",title='recovered'),
        tooltip=['recovered',"Date"]
        ).interactive()
    
    dailyvacc=alt.Chart(groupbydaterecov,width=700,height=400).mark_line().encode(
        x="Date",
        y=alt.Y("New recovered",title="New recovered"),
        tooltip=['New recovered','Date']
        ).interactive()
    fullyvalue=int(cumanali['people_fully_vaccinated'])
    dailyvaccvalue=int(cumanali['daily_vaccinations'])
    confirmedvalue=groupbydate['Confirmed'].tail(1).to_string(index=False)
    newconfirmedvalue=groupbydate['New Cases'].tail(1).to_string(index=False)
    deathvalue=groupbydate['deaths'].tail(1).to_string(index=False)
    newdeathvalue=groupbydate['New Deaths'].tail(1).to_string(index=False)
    recoveredvalue=groupbydaterecov['recovered'].tail(1).to_string(index=False)
    newrecoveredvalue=groupbydaterecov['New recovered'].tail(1).to_string(index=False)
    if selectcase == "Confirmed Cases":
        st.altair_chart(fully+vacc_daily+cofvsvacc+dailycases)
        st.write(f' 🟩 Total People Fully Vaccinated are {fullyvalue}, 🟫 daily vaccinations {dailyvaccvalue},🟥 Total Confirmed are {confirmedvalue},🟦 New Confirmed cases are {newconfirmedvalue}')
    elif selectcase == "Deaths":
        st.altair_chart(fully+ vacc_daily + deathvsvacc + dailydeath)
        st.write(f' 🟩 Total People Fully Vaccinated are {fullyvalue}, 🟫 daily vaccinations {dailyvaccvalue},🟥 Total Death are {deathvalue},🟦New Deaths  are {newdeathvalue}')
    else:
        st.altair_chart(fully+vacc_daily+recovvsvacc+dailyvacc)
        st.write(f' 🟩 Total People Fully Vaccinated are {fullyvalue}, 🟫 daily vaccinations {dailyvaccvalue},🟥 Total Recovered are {recoveredvalue},🟦New Recovered  are {newrecoveredvalue}')
     
elif my_model == "Forecasting":
    st.markdown("<h1 style='text-align: center;'>Forecasting</h1>", unsafe_allow_html=True)

    selectionbox=st.selectbox("Select Option:",["New cases forecasting","New death forecasting "])

    sliderfh=st.slider("chooes:",min_value=1,max_value=30,value=7)

    st.write("## Global Prediction of Cases")
   # if selectionbox=="New cases forecasting":
        
       #  dfcovid=covid19_DataFram[['Date','New Cases']]
        # dfcovid19=dfcovid.groupby('Date').sum()
        # futuerdf=dfcovid19.reset_index()
        # dfcovid19 = dfcovid19.asfreq('D')
        # exp = TimeSeriesExperiment()
        # exp.setup(dfcovid19, fh = sliderfh, fold = 3, session_id = 123)
         #exp.plot_model(plot = 'ts',display_format='streamlit')
        # model = exp.create_model("ets")
        # tuned_model = exp.tune_model(model, search_algorithm='grid')
         # Trains the model with the best hyperparameters on the entire dataset now
        # final_model = exp.finalize_model(tuned_model)
        # exp.plot_model(final_model,display_format='streamlit')
        # prodection=exp.predict_model(final_model,round=0)
         
        # prodection.reset_index(drop=True,inplace=True)
        # prodection=pd.DataFrame(prodection)
        # date=futuerdf['Date'].tail(1).to_string(index=False)
        # Date=pd.date_range(start=date,periods=2)
        # Date=pd.DataFrame(Date,columns=["Date"])
        # Date=Date["Date"].tail(1).to_string(index=False)
        # Date=pd.date_range(start=Date,periods=sliderfh)
        # Date=pd.DataFrame(Date,columns=["Date"])
        # prodection=prodection.join(Date['Date'])
        # prodection=prodection[['Date','New Cases']]
         #prodection['Date']=prodection['Date'].dt.date
        # prodection['New Cases']=prodection['New Cases'].astype(int)
        # prodection.index += 1
       # st.write(prodection)
         
       #  st.download_button(
        # label="Download data as CSV",
        # data=prodection.to_csv(),
         #file_name='production New cases  global.csv',
          #   )
    #else:
     #    dfcovid=covid19_DataFram[['Date','New Deaths']]
     #    dfcovid19=dfcovid.groupby('Date').sum()
     #    futuerdf=dfcovid19.reset_index()
     #    dfcovid19 = dfcovid19.asfreq('D')
     #    exp = TimeSeriesExperiment()
      #   exp.setup(dfcovid19, fh = sliderfh, fold = 3, session_id = 123)
         #exp.plot_model(plot = 'ts',display_format='streamlit')
      #   model = exp.create_model("ets")
       #  tuned_model = exp.tune_model(model, search_algorithm='grid')
         # Trains the model with the best hyperparameters on the entire dataset now
        # final_model = exp.finalize_model(tuned_model)
        # exp.plot_model(final_model,display_format='streamlit')
        # prodection=exp.predict_model(final_model,round=0)
         
        # prodection.reset_index(drop=True,inplace=True)
        # prodection=pd.DataFrame(prodection)
        # date=futuerdf['Date'].tail(1).to_string(index=False)
        # Date=pd.date_range(start=date,periods=2)
        # Date=pd.DataFrame(Date,columns=["Date"])
        # Date=Date["Date"].tail(1).to_string(index=False)
        # Date=pd.date_range(start=Date,periods=sliderfh)
        # Date=pd.DataFrame(Date,columns=["Date"])
        # prodection=prodection.join(Date['Date'])
        # prodection=prodection[['Date','New Deaths']]
        # prodection['Date']=prodection['Date'].dt.date
        # prodection['New Deaths']=prodection['New Deaths'].astype(int)
        # prodection.index += 1
        # st.write(prodection)
        # st.download_button(
        # label="Download data as CSV",
        # data=prodection.to_csv(),
         #file_name='production New death global.csv',
          #   )

    st.write("## Country-wise Prediction of Cases")
    selectionbox2=st.selectbox('Select Country: ',covid19_DataFram["Country/Region"].unique())
    dfcountryselection=covid19_DataFram[covid19_DataFram["Country/Region"]== selectionbox2]
   # if selectionbox=="New cases forecasting":
        
     #    dfcovidNewCases=dfcountryselection[['Date','New Cases']]
      #   dfcovidNewCases=dfcovidNewCases.groupby('Date').sum()
      #   futuerdf=dfcovid19.reset_index()
       #  #dfcovidNewCases=dfcovid19NewCases.asfreq('D')
        # exp = TimeSeriesExperiment()
     #    exp.setup(dfcovidNewCases, fh = sliderfh, fold = 3, session_id = 123)
      #   #exp.plot_model(plot = 'ts',display_format='streamlit')
       #  model = exp.create_model("ets")
        # tuned_model = exp.tune_model(model, search_algorithm='grid')
         # Trains the model with the best hyperparameters on the entire dataset now
       #  final_model = exp.finalize_model(tuned_model)
        # exp.plot_model(final_model,display_format='streamlit')
        # prodection=exp.predict_model(final_model,round=0)
        # prodection.reset_index(drop=True,inplace=True)
        # prodection=pd.DataFrame(prodection)
        # date=futuerdf['Date'].tail(1).to_string(index=False)
        # Date=pd.date_range(start=date,periods=2)
        # Date=pd.DataFrame(Date,columns=["Date"])
        # Date=Date["Date"].tail(1).to_string(index=False)
        # Date=pd.date_range(start=Date,periods=sliderfh)
        # Date=pd.DataFrame(Date,columns=["Date"])
        # prodection=prodection.join(Date['Date'])
        # prodection=prodection[['Date','New Cases']]
        # prodection['Date']=prodection['Date'].dt.date
        # prodection['New Cases']=prodection['New Cases'].astype(int)
        # prodection.index += 1
        # st.write(prodection)
        # st.download_button(
        # label="Download data as CSV",
        # data=prodection.to_csv(),
        # file_name=f'production New cases for {selectionbox2}.csv',
        #     )
   # else:
    #     dfcovid=dfcountryselection[['Date','New Deaths']]
    #     dfcovid19=dfcovid.groupby('Date').sum()
    #     futuerdf=dfcovid19.reset_index()
    #     dfcovid19 = dfcovid19.asfreq('D')
    #     exp = TimeSeriesExperiment()
    #     exp.setup(dfcovid19, fh = sliderfh, fold = 3, session_id = 123)
    #     #exp.plot_model(plot = 'ts',display_format='streamlit')
    #     model = exp.create_model("ets")
    #     tuned_model = exp.tune_model(model, search_algorithm='grid')
    #     # Trains the model with the best hyperparameters on the entire dataset now
    #     final_model = exp.finalize_model(tuned_model)
    #     exp.plot_model(final_model,display_format='streamlit')
    #     prodection=exp.predict_model(final_model,round=0)
    #     
    #     prodection.reset_index(drop=True,inplace=True)
    #     prodection=pd.DataFrame(prodection)
    #     date=futuerdf['Date'].tail(1).to_string(index=False)
    #     Date=pd.date_range(start=date,periods=2)
    #     Date=pd.DataFrame(Date,columns=["Date"])
    #     Date=Date["Date"].tail(1).to_string(index=False)
    #     Date=pd.date_range(start=Date,periods=sliderfh)
    #     Date=pd.DataFrame(Date,columns=["Date"])
    #     prodection=prodection.join(Date['Date'])
    #     prodection=prodection[['Date','New Deaths']]
    #     prodection['Date']=prodection['Date'].dt.date
    #     prodection['New Deaths']=prodection['New Deaths'].astype(int)
    #     prodection.index += 1
    #     st.write(prodection)
    #     st.download_button(
    #     label="Download data as CSV",
    #     data=prodection.to_csv(),
    #     file_name=f'production  New death for {selectionbox2}.csv',
    #         )
        


        

elif my_model == "Search":
    st.write("# Search by Country")

    choosecountry=st.selectbox('Search by country', covid19_DataFram["Country/Region"].unique())

    countrychosen=covid19_DataFram[covid19_DataFram["Country/Region"]== choosecountry]

    search_confirmed= alt.Chart(countrychosen,width=700,height=400).encode(
        x="Date",
        y=alt.Y("Confirmed",title="infection cases"),
        tooltip=['Confirmed']
        ).interactive()
    conf_anali=countrychosen['Confirmed'].tail(1).to_string(index=False)
    st.write(f'## Total Confirmed cases in **{choosecountry}** are **{conf_anali}**')
    st.altair_chart(search_confirmed.mark_line(color="teal"))


    sarch_newcases=alt.Chart(countrychosen,width=700,height=400).mark_point(color="lime").encode(
        x="Date",
        y="New Cases",
        tooltip=["Country/Region",'New Cases','Date']
        ).interactive()
    st.write('## Daily new Cases')
    st.altair_chart(sarch_newcases)
    st.write('# ------------------------------------------------')

    search_death= alt.Chart(countrychosen,width=700,height=400).encode(
         x="Date",
         y=alt.Y("deaths",title="Deaths cases"),
         tooltip=['deaths']
        ).interactive()
    death_anali=countrychosen['deaths'].tail(1).to_string(index=False)
    st.write(f'## Total Deaths in **{choosecountry}** are **{death_anali}**')
    st.altair_chart(search_death.mark_area(color='navy'))


    search_newdeath=alt.Chart(countrychosen,width=700,height=400).mark_point(color='turquoise').encode(
        x="Date",
        y="New Deaths",
        tooltip=["Country/Region",'New Deaths','Date']
        ).interactive()
    st.write('## Daily new deaths')
    st.altair_chart(search_newdeath)
    st.write('# ------------------------------------------------')

    #dfrecov=recov[recov["Country/Region"]== choosecountry]
    #search_recovered= alt.Chart(dfrecov,width=700,height=400).encode(
    #    x="Date",
    #    y=alt.Y("recovered",title="Recovered cases"),
    #   tooltip=['recovered']
    #    ).interactive()
    #recov_anali=dfrecov['recovered'].tail(1).to_string(index=False)
    #st.write(f'## Total recovered in **{choosecountry}** are **{recov_anali}**')
    #st.altair_chart(search_recovered.mark_area(color='salmon'))


    #search_newrecovered=alt.Chart(dfrecov,width=700,height=400).mark_line(color="purple").encode(
    #   x="Date",
    #    y="New recovered",
    #    tooltip=["Country/Region",'New recovered','Date']
    #    ).interactive()
    #st.write('## Daily new recovered')
    #st.altair_chart(search_newrecovered)
    #st.write('# ------------------------------------------------')


    chosencountryvaccin=covid_vaccin[ covid_vaccin['location']==choosecountry]

    search_vaccadmins=alt.Chart(chosencountryvaccin,width=700,height=400).encode(
        x=alt.X("date",sort='-y',title='Date',axis=alt.Axis(grid=False)),
        y=alt.Y("total_vaccinations",axis=alt.Axis(grid=False)),
        tooltip=['total_vaccinations','date']
        ).interactive()
    vaccadmins=chosencountryvaccin['total_vaccinations'].tail(1)
    v=int(vaccadmins)
    st.write(f"## Total doses administered in **{choosecountry}** are **{v}**")
    st.altair_chart(search_vaccadmins.mark_line(color='red'))

    st.write('# ------------------------------------------------')

    search_atleastonedoses= alt.Chart(chosencountryvaccin,width=700,height=400).encode(
        x=alt.X("date",sort='-y',title='Date',axis=alt.Axis(grid=False)),
        y=alt.Y("people_vaccinated",axis=alt.Axis(grid=False)),
        tooltip=['people_vaccinated','date']
        ).interactive()
    atleastonedoses=chosencountryvaccin['people_vaccinated'].tail(1)
    atleas1=int(atleastonedoses)
    st.write(f"## Total people vaccinated at least one dose in **{choosecountry}** are **{atleas1}**")
    st.altair_chart(search_atleastonedoses.mark_line(color='green'))
    st.write('# ------------------------------------------------')


    search_completed=alt.Chart(chosencountryvaccin,width=700,height=400).encode(
        x=alt.X("date",sort='-y',title='Date',axis=alt.Axis(grid=False)),
        y=alt.Y("people_fully_vaccinated",axis=alt.Axis(grid=False)),
        tooltip=['people_fully_vaccinated','date']
        ).interactive()
    fullyvacc=chosencountryvaccin['people_fully_vaccinated'].tail(1)
    fully=int(fullyvacc)
    st.write(f"## Total people fully vaccinated in **{choosecountry}** are **{fully}**")
    st.altair_chart(search_completed.mark_area(color='pink'))

    st.write('# ------------------------------------------------')

    search_bosters=alt.Chart(chosencountryvaccin,width=700,height=400).encode(
        x=alt.X("date",sort='-y',title='Date',axis=alt.Axis(grid=False)),
        y=alt.Y("total_boosters",axis=alt.Axis(grid=False)),
        tooltip=['total_boosters','date']
        ).interactive()
    bostersvacc=chosencountryvaccin['total_boosters'].tail(1)
    bos=int(bostersvacc)
    st.write(f"## Total boosters in **{choosecountry}** are **{bos}**")
    st.altair_chart(search_bosters.mark_line(color='Orange'))


    st.write('# ------------------------------------------------')

    search_dailyvacc=alt.Chart(chosencountryvaccin,width=700,height=400).encode(
        x=alt.X("date",title='Date',axis=alt.Axis(grid=False)),
        y=alt.Y("daily_vaccinations",axis=alt.Axis(grid=False)),
        tooltip=['daily_vaccinations','location','date']
        ).interactive() 
    st.write(f"## Daily vaccinations in **{choosecountry}** ") 
    st.altair_chart(search_dailyvacc.mark_line(color='green'))
    
else:   
     dropdown=st.selectbox('select your options:', ['Videos related  to Covid-19',"Search About Information", 'Web Links about Covid-19', 'References used in the Project', 'Covid question answering'])
     if dropdown =='Videos related  to Covid-19':
        st.write('### 1. Coronavirus disease (COVID-19)')
        st.video('https://www.youtube.com/watch?v=i0ZabxXmH4Y')
        st.write('### 2. COVID-19 Animation: What Happens If You Get Coronavirus?')
        st.video('https://www.youtube.com/watch?v=5DGwOJXSxqg')
        st.write('### 3. The road to a COVID-19 vaccine')
        st.video('https://www.youtube.com/watch?v=CrsnwQZIak8')
        st.write('### 4. Where did the coronavirus come from? | COVID-19 Special')
        st.video('https://www.youtube.com/watch?v=dLKMgS-WCDc')
        st.write('### 5. Pregnancy and Coronavirus (COVID-19)')
        st.video('https://www.youtube.com/watch?v=-n8C1mYca7k')
        st.write('### 6. COVID-19 coronavirus vaccine: everything you need to know')
        st.video('https://www.youtube.com/watch?v=UPkVbZ9X_jQ')
        st.write('### 7. How the COVID-19 vaccines were created so quickly')
        st.video('https://www.youtube.com/watch?v=v-NEr3KCug8')
        st.write('### 8. How to treat covid-19 | The Economist')
        st.video('https://youtu.be/jDCnaN9PXBE')
     elif dropdown == "Search About Information":
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',return_token_type_ids = True)
        model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

        context = "Corona viruses are a group of viruses that can cause diseases such as the common cold, severe acute respiratory syndrome (SARS) and Middle East respiratory syndrome (Mers). " \
                  "But, a new type of corona virus has been discovered after it was identified as the cause of the spread of a disease that began in China in 2019." \
                  "The virus is now known as ,Severe Acute Respiratory Syndrome Virus 2, and is denoted as SARS-CoV-2. The resulting disease is called coronavirus disease 2019 (COVID-19)." \
                  "In March 2020, the World Health Organization (WHO) announced that it had classified the COVID-19 pandemic as a pandemic"\
                  "Pandemic public health groups monitor and post updates online, including the US Centers for Disease Control and Prevention (CDC) and the World Health Organization (WHO)."\
                  "These groups also issued recommendations about the prevention and treatment of the disease,"\
                  "Signs and symptoms of COVID-19 may appear two to 14 days after exposure. The period after exposure to the virus and before symptoms appear is called the incubation period. " \
                  "Common signs and symptoms may include Fever, cough and feeling tired."\
                  "However, the above list does not include all symptoms, it is not exhaustive."\
                  "Also, Children have symptoms similar to those of adults and generally have some degree of illness."\
                  "Symptoms of COVID-19 can range from very mild to severe. Some people may have only a few symptoms, while others have no symptoms at all."\
                  "Some people may feel a worsening of symptoms about a week after they start, such as worsening shortness of breath and pneumonia."\
                  "But the question is, how can I protect myself from this epidemic? Get vaccinated when you qualify."\
                  "Wear masks whenever required. Currently, if you live in an area with higher rates of COVID-19 transmission,"\
                  "wear masks indoors with people outside of your immediate family, even if you have been vaccinated. Consider wearing it outdoors if you'll be in a large group of people."\
                  "Practice physical distancing (staying at least six feet away from others)."\
                  "Avoid crowds and poorly ventilated places.Wash or sanitize your hands with an alcohol-based gel frequently.Cover coughs and sneezes with the inside of your elbow."\
                  "Clean and disinfect high-touch surfaces.Monitor your health daily. If you have any symptoms related to COVID-19, stay at home except for testing for the virus."\
                  #"The best way to treat disease is prevention, so get vaccinated when you qualify. Since COVID-19 is caused by a virus, antibiotics will not work."\
                  #"Few treatments have been developed for the disease, but their effectiveness varies. Long-term complications can arise from COVID-19, and there is no known treatment yet on how to reduce this risk." \
                  #"As for the vaccine, it protects you from contracting COVID-19 or from getting serious illness or death from COVID-19Preventing the spread of COVID-19 to others"\
                  #"The number of vaccinated community members increases against COVID-19 - which slows the spread of the disease and contributes to herd immunity (so-called herd immunity) Preventing the virus that causes COVID-19"\
                  #" from spreading and replication, the two processes that allow it to form a mutation that may be better able to resist vaccines"\
                  #"Currently, there are several vaccines for COVID-19 that are undergoing clinical trials. The US Food and Drug Administration continues to evaluate the results of these trials before approving or licensing the use of Covid-19 vaccines. "\
                  #"But due to the urgent need for COVID-19 vaccines, and because the FDA approval process can take anywhere from several months to several years, the FDA initially issued an emergency use authorization for COVID-19 vaccines based on less data than is usually required. "\
                  #"Data must show that vaccines are safe and effective before the FDA can issue an emergency use approval or authorization. Vaccines with FDA approval or emergency use authorization include:"\
                  #"Pfizer-Bioentiq vaccine for COVID-19. The US Food and Drug Administration has approved the Pfizer-Bioentiq vaccine, now called Comirnaty, to prevent COVID-19 in people 16 years of age and older."\
                  #"The U.S. Food and Drug Administration approved the Comirnaty vaccine after data found it to be safe and effective. The Pfizer-Biointech vaccine is 91% effective in preventing symptoms of COVID-19 infection in people 16 years of age and older."\
                  #"Also, the Moderna vaccine for Covid 19. The Covid 19 vaccine produced by Moderna is 94% effective in preventing symptoms of Covid 19. This vaccine is approved for"\
                  #"use in persons 18 years of age and over. It takes two injections 28 days apart. The second dose may be given up to six weeks after the first dose, if needed."\
                  #"Janssen/Johnson & Johnson COVID-19 vaccine. In clinical trials, this vaccine was 66% effective in preventing symptomatic COVID-19 infection, 14 days after vaccination."\
                  #"The vaccine was also 85% effective in preventing severe COVID-19, at least "\
                  #"28 days after receiving the vaccine. This vaccine is licensed for persons 18 years of age and older. It requires one injection. The US Food and Drug Administration (FDA) and the Centers for Disease"\
                  #"Control and Prevention (CDC) have recommended continued use of this vaccine in the United States because the benefits outweigh the risks."\
                  #"If you take this vaccine, you should be educated about the potential risks and possible symptoms of a problem involving blood clotting."\
                  #"Finally, if you have any signs of an allergic reaction, seek help immediately. Tell your doctor about your allergic reaction, even if it goes away on its own or if you don't get emergency care." \
                  #"This reaction may mean that you are allergic to the vaccine. You may not be able to get a "\
                  #"second dose of the same vaccine. However, you may be able to get a different type of vaccine when you take the second dose."
                      
        question =st.text_input("Enter Your Question") 

        encoding = tokenizer.encode_plus(question, context)


        input_ids= encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        start_scores, end_scores = model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))

        ans_tokens = input_ids[torch.argmax(start_scores) : torch.argmax(end_scores)+1]
        answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens , skip_special_tokens=True)

        #st.write("\nQuestion ",question) 
        #st.write("\nAnswer Tokens: ")
        #st.write(answer_tokens)

        answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)

        #st.write("### \nAnswer : ",answer_tokens_to_string)

        if answer_tokens:
            st.write("### \nAnswer : ",answer_tokens_to_string)
        elif question == "":
            st.write("### \nAnswer : ")
        else:
            st.write("##  ⚠️ It seems that there are no results matching your search please try again⚠️❗ ")
            
            
            
         
     elif dropdown == 'Web Links about Covid-19':
        st.write("### 1. Statistics [click here](https://www.worldometers.info/coronavirus/)")
        st.write("### 2. COVID-19 Global [click here](https://reliefweb.int/topics/covid-19-global?gclid=Cj0KCQiAys2MBhDOARIsAFf1D1dtvs2Hl3u7Bha7WyCI6Mnz3Jf_izcje05ONRDOyAcoeSRAt-HFONMaAnMWEALw_wcB)")
        st.write("### 3. WHO Coronavirus (COVID-19) Dashboard [click here](https://covid19.who.int/)")
        st.write("### 4. The most frequently question about covid 19 [click here](https://www.gavi.org/vaccineswork/covid-19-faqs-10-questions-you-need-know-answer?gclid=Cj0KCQiAys2MBhDOARIsAFf1D1dhF7DqNCCo1__ZwneZnWB8yRWbDHSYsRXWQWniEuniBxJYsvgresEaAvdOEALw_wcB)")
        st.write("### 5. Everything you need to know about covid 19 [click here](https://www.gavi.org/covid19-vaccines?gclid=Cj0KCQiAys2MBhDOARIsAFf1D1e0mBsmJhLEW-GAXQnMRnB-fz4VTYsQUTgcGkwPWBaHM5xTRoH9sZcaAnvMEALw_wcB)")
        st.write("### 6. Covid 19 statistics and research [click here](https://ourworldindata.org/coronavirus)")
        st.write('### 7. Covid 19 data tracker [click here](https://covid.cdc.gov/covid-data-tracker/#datatracker-home)')
        st.write('### 8. Covid 19 vaccination tracker [click here](https://www.pharmaceutical-technology.com/covid-19-vaccination-tracker)')
        
     elif dropdown == 'References used in the Project':
        st.write("## Datasets References")
        st.write("1. COVID-19/time_series_covid19_confirmed_global.csv at master .... (n.d.). Retrieved November 24, 2021, from https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv.")
        st.write("2. COVID-19/time_series_covid19_deaths_global.csv at master .... (n.d.). Retrieved November 24, 2021, from https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv.")
        st.write("3. COVID-19/time_series_covid19_recovered_global.csv at master .... (n.d.). Retrieved November 24, 2021, from https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv.")
        st.write("4. covid-19-data/vaccinations.csv at master · owid/covid-19-data · GitHub. (n.d.). Retrieved November 24, 2021, from https://github.com/owid/covid-19-data/blob/master/public/data/vaccinations/vaccinations.csv.")
        st.write("5. covid-19-data/owid-covid-codebook.csv at master. (n.d.). Retrieved December 11, 2021, from https://github.com/owid/covid-19-data/blob/master/public/data/owid-covid-codebook.csv.")
        st.write("6. jpldgالجريدة الرسمية قانون القيمة المضافة. (n.d.). Retrieved December 11, 2021, from https://aliijazat-alrasmia.netlify.app/jpldg%D8%A7%D9%84%D8%AC%D8%B1%D9%8A%D8%AF%D8%A9-%D8%A7%D9%84%D8%B1%D8%B3%D9%85%D9%8A%D8%A9-%D9%82%D8%A7%D9%86%D9%88%D9%86-%D8%A7%D9%84%D9%82%D9%8A%D9%85%D8%A9-%D8%A7%D9%84%D9%85%D8%B6%D8%A7%D9%81%D8%A9.html.")
        st.write("## Other References")
        st.write('1. In-text: (Värri, Delgado & Gallos, 2020)Your Bibliography: Värri, A., Delgado, J., & Gallos, P. (2020). Integrated Citizen Centered Digital Health and Social Care. IOS Press, Incorporated.')
        st.write('2. Blogathon, D. (2021). COVID-19 Dashboard in Python using Streamlit. Retrieved 5 March 2021, from https://medium.com/analytics-vidhya/covid-19-dashboard-in-python-using-streamlit-aa58581e5a7f')
        st.write('3. Philp, R. (2021). 10 Tips for Visualizing COVID-19 Data - Global Investigative Journalism Network. Retrieved 5 March 2021, from https://gijn.org/2020/06/18/10-tips-for-visualizing-covid-19-data/')
        st.write('4. Boka, D., & Wainer, H. (2020). How Can We Estimate the Death Toll from COVID-19? CHANCE, 33(3), 67-72. doi: 10.1080/09332480.2020.1787743')
        st.write('5. Jordana, J., & Triviño-Salazar, J. (2020). Where are the ECDC and the EU-wide responses in the COVID-19 pandemic?. The Lancet, 395(10237), 1611-1612. doi: 10.1016/s0140-6736(20)31132-6')
        st.write('6. Tzanou, M. Health data privacy under the GDPR')
        st.write('7. Sheng, J., Amankwah‐Amoah, J., Khan, Z., & Wang, X. (2020). COVID‐19 Pandemic in the New Era of Big Data Analytics: Methodological Innovations and Future Research Directions. British Journal Of Management. doi: 10.1111/1467-8551.12441')
        st.write('8. Trung, L. (2019). https://euroasia-science.ru/pdf-arxiv/the-controllability-function-of-polynomial-for-descriptor-systems-23-31/. Eurasianunionscientists, 4(65). doi: 10.31618/esu.2413-9335.2019.4.65.275')
        st.write('9. Sheng, J., Amankwah‐Amoah, J., Khan, Z., & Wang, X. (2020). COVID‐19 Pandemic in the New Era of Big Data Analytics: Methodological Innovations and Future Research Directions. British Journal of Management')
        st.write('10. Chen, E., Lerman, K., & Ferrara, E. (2020). Tracking social media discourse about the covid-19 pandemic: Development of a public coronavirus twitter data set. JMIR Public Health and Surveillance, 6(2), e19273')
        st.write('11. Gallotti, R., Valle, F., Castaldo, N., Sacco, P., & De Domenico, M. (2020). Assessing the risks of ‘infodemics’ in response to COVID-19 epidemics. Nature Human Behaviour, 4(12), 1285-1293')
        st.write('12. Shinde, G., Kalamkar, A., Mahalle, P., Dey, N., Chaki, J., & Hassanien, A. (2020). Forecasting Models for Coronavirus Disease (COVID-19): A Survey of the State-of-the-Art. SN Computer Science, 1(4). doi: 10.1007/s42979-020-00209-9')
        st.write('13. CDC COVID Data Tracker. (n.d.). Retrieved November 16, 2021, from https://covid.cdc.gov/covid-data-tracker.')
        st.write('14. COVID Live Update: 254,728,689 Cases and 5,125,684 Deaths from .... (n.d.). Retrieved November 16, 2021, from https://www.worldometers.info/coronavirus.')
        st.write('15. WHO Coronavirus (COVID-19) Dashboard. (n.d.). Retrieved November 16, 2021, from https://covid19.who.int.')
        st.write('16. Coronavirus Pandemic (COVID-19) - Statistics and Research - Our .... (n.d.). Retrieved November 16, 2021, from https://ourworldindata.org/coronavirus.')
        st.write('17. COVID-19 Vaccination Tracker: Daily Rates, Statistics & Updates. (n.d.). Retrieved November 16, 2021, from https://www.pharmaceutical-technology.com/covid-19-vaccination-tracker.')
        st.write('18. A Beginner s Guide to Time Series Modelling Using PyCaret. (n.d.). Retrieved December 11, 2021, from https://analyticsindiamag.com/a-beginners-guide-to-time-series-modelling-using-pycaret.')
        st.write("19. New Time Series Forecasting with PyCaret! | Towards Data Science. (n.d.). Retrieved December 11, 2021, from https://towardsdatascience.com/new-time-series-with-pycaret-4e8ce347556a.")
        st.write("20. PyCaret. Automate key steps to evaluate and… | MLearning.ai. (n.d.). Retrieved December 11, 2021, from https://medium.com/mlearning-ai/pycaret-73b519e2d4d6.")
        st.write("21. Announcing PyCaret's New Time Series Module | by Moez Ali | Nov .... (n.d.). Retrieved December 11, 2021, from https://towardsdatascience.com/announcing-pycarets-new-time-series-module-b6e724d4636c.")
     else:
         st.write('### What is coronavirus?')
         st.write('Coronaviruses are a family of viruses that can cause respiratory illness in humans. They are called “corona” because of crown-like spikes on the surface of the virus. Severe acute respiratory syndrome (SARS), Middle East respiratory syndrome (MERS) and the common cold are examples of coronaviruses that cause illness in humans.The new strain of coronavirus — COVID-19 — was first reported in Wuhan, China in December 2019. The virus has since spread to all continents.')
         st.write('### How do you get infected with COVID-19?')
         st.write('COVID-19 enters your body through your mouth, nose or eyes (directly from the airborne droplets or from transfer of the virus from your hands to your face). The virus travels to the back of your nasal passages and mucous membrane in the back of your throat. It attaches to cells there, begins to multiply and moves into lung tissue. From there, the virus can spread to other body tissues.')
         st.write('### How does the new coronavirus (COVID-19) spread from person to person?')
         st.write('COVID-19 is likely spread:')
         st.write('•	The virus travels in respiratory droplets released into the air when an infected person coughs, sneezes, talks, sings or breathes near you (within 6 feet). You may be infected if you inhale these droplets.')
         st.write('•	You can also get COVID-19 from close contact (touching, shaking hands) with an infected person and then touching your face.')
         st.write('•	It’s considered possible to get COVID-19 after touching a contaminated surface and then touching your eyes, mouth, or nose before washing your hands. But it’s thought to be unlikely.')
         st.write('### Where do coronaviruses come from?')
         st.write('Coronaviruses are often found in bats, cats and camels. The viruses live in but do not infect the animals. Sometimes these viruses then spread to different animal species. The viruses may change (mutate) as they transfer to other species. Eventually, the virus can jump from animal species and begins to infect humans. In the case of COVID-19, the first people infected in Wuhan, China are thought to have contracted the virus at a food market that sold meat, fish and live animals. Although researchers don’t know exactly how people were infected, they already have evidence that the virus can be spread directly from person to person through close contact.')
         st.write('### What’s different about the delta variant of COVID-19?')
         st.write('It’s normal for viruses to mutate — especially coronaviruses and influenza viruses. These mutations create new variants of the virus. Sometimes the variants are less contagious, less severe or have slightly different presenting symptoms. Unfortunately, the delta variant of COVID-19 (a strain called B.1.617.2) is more highly contagious and more likely to result in severe illness.')
         st.write('### How long is a person infected with COVID-19 considered contagious?')
         st.write('If you’re infected with COVID-19 it can take several days to develop symptoms — but you are contagious during this time. You are no longer contagious 10 days after your symptoms began.')
         st.write('The best way to avoid spreading COVID-19 to others is to:')
         st.write('•	Stay 6 feet away from others whenever possible.')
         st.write('•	Wear a cloth mask that covers your mouth and nose when around others.')
         st.write('•	Wash your hands often. If soap is not available, use a hand sanitizer that contains at least 60% alcohol.')
         st.write('•	Avoid crowded indoor spaces. Bring in outdoor air as much as possible.')
         st.write('•	Stay self-isolated at home if you are feeling ill with symptoms that could be COVID-19 or have a positive test for COVID-19.')
         st.write('•	Clean and disinfect frequently touched surfaces.')
         st.write('### How soon after becoming infected with COVID-19 will I develop symptoms?')
         st.write('The time between becoming infected and showing symptoms (incubation period) can range from 2 to 14 days. The average time before experiencing symptoms is five days. Symptoms can range in severity from very mild to severe. In about 80% of patients, COVID-19 causes only mild symptoms, although this may change as variants emerge.')
         st.write('### How to deal with a covid 19 situation ?')
         st.write(' 1. Take a test')
         st.write('If someone in your household has tested positive for COVID-19, Dr. Scott Braunstein, medical director of Sollis Health in Los Angeles, recommends getting a COVID-19 PCR test for yourself as soon as feasible.')
         st.write('2. Limit contact')
         st.write('In its principles,The Centers for Disease Control and Prevention (CDC), a trusted source for caring for someone with COVID-19, states: COVID-19 sufferers should isolate themselves from the rest of the household. If possible, the COVID-19 patient should have his or her own bedroom and bathroom. Everyone in the house should keep at least 6 feet away from the COVID-1 sufferer.')
         st.write('3. Leave the windows open')
         st.write('Keep the windows open as often as possible to increase ventilation in your living quarters, even if it means turning up the heat.')
         st.write('4. Put on a mask')
         st.write('According to the CDC, anyone who is sick should wear a mask or face covering while they are around other people, and anyone living with them should wear a mask or face covering as well.')
         st.write('5. Hands should be washed, and surfaces should be disinfected.')
         st.write('The CDC recommends that you keep your hands clean:')
         st.write('Hands should be washed often with soap and water for at least 20 seconds, especially after being near a sick person.')
        
