import requests
import bs4
import pandas as pd
import re

def get_city_locations():
    url = "https://astronomia.zagan.pl/art/wspolrzedne.html"
    line_regex = r"(.+?)\s+(\d{2})°(\d{2}).{2}\s+?(\d{2})°(\d{2}).{2}"

    r = requests.get(url)
    raw_data = r.content.decode("iso-8859-2")

    data_lines = re.findall(line_regex, raw_data)
    df = pd.DataFrame(data_lines, columns=["name", "long_min", "long_sec", "lat_min", "lat_sec"])
    df["longitude"]  = df["long_min"].astype(int) + df["long_sec"].astype(int) / 60
    df["latitude"] = df["lat_min"].astype(int) + df["lat_sec"].astype(int) / 60
    df = df[["name", "longitude", "latitude"]]

    return df

def get_city_population():
    # Wikipedia spis ludności
    url = "https://pl.wikipedia.org/wiki/Dane_statystyczne_o_miastach_w_Polsce"
    tabels = pd.read_html(url)
    main_table = tabels[0]
    main_table.columns = ["name", "region1", "region2", "area", "population", "population_density"]
    return main_table

city_df = get_city_locations()
population_df = get_city_population()

joined = city_df.set_index("name").join(
    population_df.set_index("name"),
    how="left"
).reset_index()
joined = joined[["name", "population", "area", "longitude", "latitude"]]
joined = joined.dropna()
joined = joined.reset_index(drop=True)

prectentage = 100 * joined["population"].sum() / population_df['population'].sum()
print(f"Pokrywamy: {prectentage:.2f}% populacji miastowej opisanej w rejestrze Polski z 2021")

print("Największe miasta:")
print(joined.sort_values("population", ascending=False).head(20))

joined.to_csv("cities.csv", index=False, encoding="utf-8")
