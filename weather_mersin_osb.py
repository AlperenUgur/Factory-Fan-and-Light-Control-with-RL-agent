from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from meteostat import Point, Daily

# Mersin OSB için sabit nokta (yaklaşık)
LAT = 36.897835
LON = 34.794194

def fetch_one_year_daily_temps():
    end = datetime.now()
    start = datetime(end.year - 1, end.month, end.day)

    p = Point(LAT, LON)
    data = Daily(p, start, end).fetch()

    # Günlük ortalama sıcaklık (°C): tavg
    # Bazı günler boş olabilir, dropna ile temizliyoruz
    temps = data["tavg"].dropna()

    return temps

def save_year_plot(temps, filename="outside_temperature_year.png"):
    plt.figure(figsize=(10, 4))
    plt.plot(temps.index, temps.values)
    plt.title("Mersin OSB - Son 1 Yıl Günlük Ortalama Sıcaklık (°C)")
    plt.xlabel("Tarih")
    plt.ylabel("Sıcaklık (°C)")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

if __name__ == "__main__":
    temps = fetch_one_year_daily_temps()

    # CSV kaydı
    temps.to_csv("outside_temperature_year.csv", header=["tavg"])
    save_year_plot(temps)

    print("Oluşturuldu:")
    print(" - outside_temperature_year.csv")
    print(" - outside_temperature_year.png")
