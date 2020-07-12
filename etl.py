import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, year, month, dayofmonth, round, regexp_replace
from pyspark.sql.types import IntegerType, FloatType
from data_dictionary import i94_state_codes, i94_visa_codes, i94_country_codes

spark = SparkSession.builder.\
    config("spark.jars.packages","saurfang:spark-sas7bdat:2.0.0-s_2.11").\
    enableHiveSupport().getOrCreate()


# Create demographics dataframe
df_demographics = spark.read.csv("us-cities-demographics.csv", sep=';', header=True)


# Convert speicficed columns to integer type for summing capability
df_demographics = df_demographics.withColumn("Count", col("Count").cast(IntegerType())) \
.withColumn("Total Population", col("Total Population").cast(IntegerType())) \
.withColumn("Male Population", col("Male Population").cast(IntegerType())) \
.withColumn("Female Population", col("Female Population").cast(IntegerType())) \
.withColumn("Foreign-born", col("Foreign-born").cast(IntegerType()))


# Create dataframe containing population sums
df_population_sum = df_demographics.groupBy("State Code").sum()


# Create dataframe by pivoting  race column and tranform rows values to individual columns
# Sum the races with the same state code
df_race = df_demographics.groupBy("State Code").pivot("Race").sum("Count").orderBy("State Code")


# Join race and population dataframes to create final US demographics by state dataframe
# Drop duplicates
df_us_demographics = df_population_sum.join(df_race, "State Code").dropDuplicates()


# Fix column names for correct format for parquet files
df_us_demographics = df_us_demographics.select(col('State Code').alias('state_code'),
                                               col('sum(Male Population)').alias('male_population'),
                                               col('sum(Female Population)').alias('female_population'), 
                                               col('sum(Total Population)').alias('total_population'),
                                               col('sum(Foreign-born)').alias('foreign_born'),
                                               col('American Indian and Alaska Native').alias('american_indian_or_alaska_native'),
                                               col('Asian').alias('asian'),
                                               col('Black or African-American').alias('black_or_african_american'),
                                               col('Hispanic or Latino').alias('hispanic_or_latino'),
                                               col('White').alias('white')
                                            )


# Parquet demographics
df_us_demographics.write.mode('overwrite').parquet("us_demographics_by_state")


# Create temperature dataframe
df_temperature = spark.read.csv("GlobalLandTemperaturesByState.csv", header=True)


# Filter the temperature by US only  
df_us_temperature = df_temperature.filter(df_temperature["Country"] == "United States")


# Reverse key value in i94addr_codes where key = full name and value = abbrev
state_code_dict = dict((v, k) for k, v in i94_state_codes.items())


# Remove (State) from Georgia 
df_us_temperature = df_us_temperature.withColumn('State', regexp_replace('State', 'Georgia \(State\)', 'Georgia'))


 # Define full name conversion to abbreviation udf 
state_abbrev_udf = udf(lambda x: state_code_dict[x])


# Break down datetime to year, month
# Convert state names to state codes
# Convert Celcius to Fahrenheit
df_us_temperature = df_us_temperature.withColumn("year", year(df_us_temperature["dt"])) \
.withColumn("month", month(df_us_temperature["dt"])) \
.withColumn("state_code", state_abbrev_udf(df_us_temperature["State"])) \
.withColumn("average_temperature_fahrenheit",round(df_us_temperature["AverageTemperature"]*(9/5)+32,1))


# Filter by year 2013, the most recent data
df_us_temperature_2013 = df_us_temperature.filter(year(df_us_temperature["dt"]) == 2013)


# Drop duplicates
# Pick columns and format for parquet
df_us_temperature_2013 = df_us_temperature_2013.select("year", "month", "state_code", "average_temperature_fahrenheit").dropDuplicates()


# Parquet temperature
df_us_temperature_2013.write.mode('overwrite').parquet("us_temperature_2013")


# Create airport data dataframe
df_airport_codes = spark.read.csv("airport-codes_csv.csv", header=True)


# Filter iso_region code by US only
df_airport_codes = df_airport_codes.filter(df_airport_codes["iso_region"].contains("US-"))


# Remove iso region 'US-U-A'
df_airport_codes = df_airport_codes.filter(df_airport_codes["iso_region"] != "US-U-A")


# Create udf to remove "US-" from iso_region to transform to "state_code"
get_state = udf(lambda x: x.replace("US-",""))


# Convert elevation type to float
df_airport_codes = df_airport_codes.withColumn("elevation_ft", col("elevation_ft").cast(FloatType()))\
.withColumn("iso_region", get_state(col("iso_region")))


# Average elevation
df_us_elevations = df_airport_codes.groupBy("iso_region").avg("elevation_ft").orderBy("iso_region")


# Round values two decimal places
df_us_elevations = df_us_elevations.select(col("iso_region").alias("state_code"),\
                                        round(col("avg(elevation_ft)"),2).alias("avg_elevation_ft"))


# Parquet elevation
df_us_elevations.write.mode('overwrite').parquet("us_elevations_by_state")


# Create immigration dataframe
df_immigration = spark.read.parquet("sas_data")


# Select columns for dataframe 
df_immigration = df_immigration.select(col("i94yr").alias("i94_year"),
                                       col("i94mon").alias("i94_month"),
                                       col("i94res").alias("i94_residence"),
                                       col("i94addr").alias("i94_address"),
                                       col("i94visa").alias("i94_visa"),
                                       "gender",
                                       col("biryear").alias("birth_year")
                                      )


# Create state code abbreviations dataframe
i94_state_codes_list = list(map(list, i94_state_codes.items()))
df_i94_state_codes = spark.createDataFrame(i94_state_codes_list, ["state_code", "state_name"])

# Create country code abbreviations dataframe
i94_country_codes_list = list(map(list, i94_country_codes.items()))
df_i94_country_codes = spark.createDataFrame(i94_country_codes_list, ["country_code", "country_name"])


# Create visa code abbreviations dataframe
i94_visa_codes_list = list(map(list, i94_visa_codes.items()))
df_i94_visa_codes = spark.createDataFrame(i94_visa_codes_list, ["visa_code", "visa_type"])


# Parquet all immigration, state, country, and visa dataframes 
df_immigration.write.mode('overwrite').partitionBy("i94_year", "i94_month").parquet("immigration")
df_i94_state_codes.write.mode('overwrite').parquet("state_codes")
df_i94_country_codes.write.mode('overwrite').parquet("country_codes")
df_i94_visa_codes.write.mode('overwrite').parquet("visa_codes")
