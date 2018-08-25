#######files downloaded from: https://www.esrl.noaa.gov/psd/cgi-bin/db_search/DBListFiles.pl?did=195&tid=67126&vid=1274

import pandas as pd
import xarray as xr

def GetClimate4mr():
    #import city lat/lon data
    mr = pd.read_csv('/Users/jamestamerius/Dropbox/Documents/Projects/EPA_Flu/EPA_SummerFlu/CityData/CityMR')
    mr['lon2'] = mr.longitude.mod(360)

    #import climate datasets
    t16 = xr.open_dataset('/Users/jamestamerius/Dropbox/Documents/Projects/EPA_Flu/EPA_SummerFlu/ClimateData/air.2m.gauss.2016.nc')
    t17 = xr.open_dataset('/Users/jamestamerius/Dropbox/Documents/Projects/EPA_Flu/EPA_SummerFlu/ClimateData/air.2m.gauss.2017.nc')
    t18 = xr.open_dataset('/Users/jamestamerius/Dropbox/Documents/Projects/EPA_Flu/EPA_SummerFlu/ClimateData/air.2m.gauss.2018.nc')

    sh16 = xr.open_dataset('/Users/jamestamerius/Dropbox/Documents/Projects/EPA_Flu/EPA_SummerFlu/ClimateData/shum.2m.gauss.2016.nc')
    sh17 = xr.open_dataset('/Users/jamestamerius/Dropbox/Documents/Projects/EPA_Flu/EPA_SummerFlu/ClimateData/shum.2m.gauss.2017.nc')
    sh18 = xr.open_dataset('/Users/jamestamerius/Dropbox/Documents/Projects/EPA_Flu/EPA_SummerFlu/ClimateData/shum.2m.gauss.2018.nc')

    #Temperature
    time16 = t16['time'].values
    time17 = t17['time'].values
    time18 = t18['time'].values
    T16 = pd.DataFrame(columns = ['time', 'temp', 'mr'])
    T17 = pd.DataFrame(columns = ['time', 'temp', 'mr'])
    T18 = pd.DataFrame(columns = ['time', 'temp', 'mr'])
    for i,row in mr.iterrows():
        temp16 = pd.DataFrame(t16.sel(lon=row.longitude, lat=row.CityLat, method='nearest')['air'].values,columns=['temp'])
        temp16['mr'] = row.mr
        temp16['time']=time16
        T16 = pd.concat([T16,temp16])

        temp17 = pd.DataFrame(t17.sel(lon=row.longitude, lat=row.CityLat, method='nearest')['air'].values,columns=['temp'])
        temp17['mr'] = row.mr
        temp17['time']= time17
        T17 = pd.concat([T17,temp17])

        temp18 = pd.DataFrame(t18.sel(lon=row.longitude, lat=row.CityLat, method='nearest')['air'].values,columns=['temp'])
        temp18['mr'] = row.mr
        temp18['time']=time18
        T18 = pd.concat([T18,temp17])

    T = pd.concat([T16,T17,T18])
    T['temp']=T['temp']-273.15

    #Specific Humidity
    time16 = sh16['time'].values
    time17 = sh17['time'].values
    time18 = sh18['time'].values
    SH16 = pd.DataFrame(columns = ['time', 'sh', 'mr'])
    SH17 = pd.DataFrame(columns = ['time', 'sh', 'mr'])
    SH18 = pd.DataFrame(columns = ['time', 'sh', 'mr'])
    for i,row in mr.iterrows():
        s16 = pd.DataFrame(sh16.sel(lon=row.longitude, lat=row.latitude, method='nearest')['shum'].values,columns=['sh'])
        s16['mr'] = row.mr
        s16['time']=time16
        SH16 = pd.concat([SH16,s16])

        s17 = pd.DataFrame(sh17.sel(lon=row.longitude, lat=row.latitude, method='nearest')['shum'].values, columns=['sh'])
        s17['mr'] = row.mr
        s17['time'] = time17
        SH17 = pd.concat([SH17, s17])

        s18 = pd.DataFrame(sh18.sel(lon=row.longitude, lat=row.latitude, method='nearest')['shum'].values, columns=['sh'])
        s18['mr'] = row.mr
        s18['time'] = time18
        SH18 = pd.concat([SH18, s18])

    SH = pd.concat([SH16,SH17,SH18])

    Climate = pd.merge(T, SH, on=['mr', 'time'])

    Climate.to_pickle('/Users/jamestamerius/Dropbox/Documents/Projects/EPA_Flu/EPA_SummerFlu/ClimateData/mrClimateData')


def GetClimate4metro():
    #import city lat/lon data
    mr = pd.read_csv('/Users/jamestamerius/Dropbox/Documents/Projects/EPA_Flu/EPA_SummerFlu/CityData/MetroLatLon')
    mr['CityLon'] = mr['CityLon'].mod(360)

    #import climate datasets
    t16 = xr.open_dataset('/Users/jamestamerius/Dropbox/Documents/Projects/EPA_Flu/EPA_SummerFlu/ClimateData/air.2m.gauss.2016.nc')
    t17 = xr.open_dataset('/Users/jamestamerius/Dropbox/Documents/Projects/EPA_Flu/EPA_SummerFlu/ClimateData/air.2m.gauss.2017.nc')
    t18 = xr.open_dataset('/Users/jamestamerius/Dropbox/Documents/Projects/EPA_Flu/EPA_SummerFlu/ClimateData/air.2m.gauss.2018.nc')

    sh16 = xr.open_dataset('/Users/jamestamerius/Dropbox/Documents/Projects/EPA_Flu/EPA_SummerFlu/ClimateData/shum.2m.gauss.2016.nc')
    sh17 = xr.open_dataset('/Users/jamestamerius/Dropbox/Documents/Projects/EPA_Flu/EPA_SummerFlu/ClimateData/shum.2m.gauss.2017.nc')
    sh18 = xr.open_dataset('/Users/jamestamerius/Dropbox/Documents/Projects/EPA_Flu/EPA_SummerFlu/ClimateData/shum.2m.gauss.2018.nc')

    #Temperature
    time16 = t16['time'].values
    time17 = t17['time'].values
    time18 = t18['time'].values
    T16 = pd.DataFrame(columns = ['time', 'temp', 'metro'])
    T17 = pd.DataFrame(columns = ['time', 'temp', 'metro'])
    T18 = pd.DataFrame(columns = ['time', 'temp', 'metro'])
    for i,row in mr.iterrows():
        temp16 = pd.DataFrame(t16.sel(lon=row.CityLon, lat=row.CityLat, method='nearest')['air'].values,columns=['temp'])
        temp16['metro'] = row.NAME
        temp16['time']=time16
        T16 = pd.concat([T16,temp16])

        temp17 = pd.DataFrame(t17.sel(lon=row.CityLon, lat=row.CityLat, method='nearest')['air'].values,columns=['temp'])
        temp17['metro'] = row.NAME
        temp17['time']= time17
        T17 = pd.concat([T17,temp17])

        temp18 = pd.DataFrame(t18.sel(lon=row.CityLon, lat=row.CityLat, method='nearest')['air'].values,columns=['temp'])
        temp18['metro'] = row.NAME
        temp18['time']=time18
        T18 = pd.concat([T18,temp17])

    T = pd.concat([T16,T17,T18])
    T['temp']=T['temp']-273.15

    #Specific Humidity
    time16 = sh16['time'].values
    time17 = sh17['time'].values
    time18 = sh18['time'].values
    SH16 = pd.DataFrame(columns = ['time', 'sh', 'metro'])
    SH17 = pd.DataFrame(columns = ['time', 'sh', 'metro'])
    SH18 = pd.DataFrame(columns = ['time', 'sh', 'metro'])
    for i,row in mr.iterrows():
        s16 = pd.DataFrame(sh16.sel(lon=row.CityLon, lat=row.CityLat, method='nearest')['shum'].values,columns=['sh'])
        s16['metro'] = row.NAME
        s16['time']=time16
        SH16 = pd.concat([SH16,s16])

        s17 = pd.DataFrame(sh17.sel(lon=row.CityLon, lat=row.CityLat, method='nearest')['shum'].values, columns=['sh'])
        s17['metro'] = row.NAME
        s17['time'] = time17
        SH17 = pd.concat([SH17, s17])

        s18 = pd.DataFrame(sh18.sel(lon=row.CityLon, lat=row.CityLat, method='nearest')['shum'].values, columns=['sh'])
        s18['metro'] = row.NAME
        s18['time'] = time18
        SH18 = pd.concat([SH18, s18])

    SH = pd.concat([SH16,SH17,SH18])

    Climate = pd.merge(T, SH, on=['metro', 'time'])

    Climate.to_pickle('/Users/jamestamerius/Dropbox/Documents/Projects/EPA_Flu/EPA_SummerFlu/ClimateData/metroClimateData')


GetClimate4metro()








