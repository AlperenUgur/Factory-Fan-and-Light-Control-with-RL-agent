import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import imageio.v2 as imageio

# =======================
# 1) 4 ODANIN KONUMU
# =======================

ROOM_POSITIONS = [
    (1, 1),  # Oda 1
    (1, 4),  # Oda 2
    (4, 1),  # Oda 3
    (4, 4),  # Oda 4
]

# İlk iki oda "sürekli ısınan" oda olsun (fan yoksa ısı yükseliyor)
HOT_ROOM_FLAGS = [True, True, False, False]


# ============================================
# 2) Tek Oda İçin RL Ortamı
# ============================================

class FactoryRoomEnv:
    """
    Tek bir oda için pekiştirmeli öğrenme ortamı.
    Durum:
        activity: 0-1
        temp: 0=Soğuk, 1=Konfor, 2=Sıcak
        hour: 0-1
        price: 0-1
    Eylemler: 0..5 ışık/fan kombinasyonları

    hot_room = True ise:
        Fan çalışmıyorsa ortamın sıcaklığı yukarı doğru tırmanmaya eğilimli.
    """

    def __init__(self, hot_room=False):
        self.n_actions = 6
        self.state = None
        self.hot_room = hot_room  # 🔥 bu oda ısınan oda mı?

    def reset(self):
        self.state = (
            random.randint(0, 1),  # activity
            random.randint(0, 2),  # temp
            random.randint(0, 1),  # hour
            random.randint(0, 1),  # price
        )
        return self.state

    def step(self, action):
        activity, temp, hour, price = self.state

        # Action mapping
        mapping = {
            0: (0, 0),  # kapalı
            1: (1, 0),  # ışık
            2: (0, 1),  # fan düşük
            3: (1, 1),  # ışık + fan düşük
            4: (0, 2),  # fan yüksek
            5: (1, 2),  # ışık + fan yüksek
        }
        light_on, fan_level = mapping[action]

        reward = 0.0

        # === ÖDÜL FONKSİYONU ===
        if activity == 1:
            if light_on:
                reward += 3   # aktifken ışık açık
            if temp == 2 and fan_level > 0:
                reward += 3   # sıcak ortamda fan konfor sağlıyor
        else:
            if light_on or fan_level > 0:
                reward -= 3   # boşken açık bırakırsan ceza

        # Pahalı saatte gereksiz kapalı kalabilirse bonus
        if price == 1 and activity == 0 and light_on == 0 and fan_level == 0:
            reward += 2

        # Enerji maliyeti
        power = 0.0
        if light_on:
            power += 1.0
        if fan_level == 1:
            power += 1.5
        elif fan_level == 2:
            power += 3.0

        cost = power * (1 if price == 0 else 2)  # pahalıysa ×2
        reward -= cost * 0.4  # biraz daha yumuşattım (0.5→0.4)

        # === SICAKLIK GÜNCELLEMESİ ===
        new_temp = temp

        # Fan çalışıyorsa soğutma etkisi
        if fan_level == 1:       # düşük fan
            if temp == 2:
                new_temp = 1
        elif fan_level == 2:     # yüksek fan
            if temp == 2:
                new_temp = 1
            elif temp == 1:
                new_temp = 0

        # Fan kapalıysa ortamın pasif davranışı
        if fan_level == 0:
            if self.hot_room:
                # 🔥 Sıcak oda: fan yoksa sürekli ısınmaya meyilli
                if new_temp < 2 and random.random() < 0.8:
                    new_temp += 1
            else:
                # Normal oda: gündüz hafif ısınma, gece hafif soğuma
                if hour == 0 and new_temp < 2 and random.random() < 0.4:
                    new_temp += 1
                elif hour == 1 and new_temp > 0 and random.random() < 0.4:
                    new_temp -= 1

        # === SAAT & FİYAT GÜNCELLEMESİ ===
        # Saat değişimi
        if random.random() < 0.1:
            hour = 1 - hour

        # Saat -> fiyat
        price = 1 if hour == 0 else 0  # gündüz pahalı, gece ucuz

        # Aktivite değişimi
        if random.random() < 0.3:
            activity = 1 - activity

        self.state = (activity, new_temp, hour, price)
        info = {"energy_cost": cost}
        return self.state, reward, False, info


# ===============================
# 3) Q Learning
# ===============================

def state_to_index(state):
    a, t, h, p = state
    # 2 * 3 * 2 * 2 = 24
    return a * 12 + t * 4 + h * 2 + p


def train_room(n_episodes=2000, max_steps=40, hot_room=False):
    env = FactoryRoomEnv(hot_room=hot_room)
    Q = np.zeros((24, 6))

    temp_hist = []
    cost_hist = []

    for ep in range(n_episodes):
        s = env.reset()
        epsilon = 1 - ep / n_episodes  # lineer azalan

        for step in range(max_steps):
            if random.random() < epsilon:
                a = random.randint(0, 5)
            else:
                a = int(np.argmax(Q[state_to_index(s)]))

            ns, r, _, info = env.step(a)

            idx_s = state_to_index(s)
            idx_ns = state_to_index(ns)

            Q[idx_s, a] += 0.1 * (r + 0.95 * np.max(Q[idx_ns]) - Q[idx_s, a])

            # Son epizot için log alıyoruz
            if ep == n_episodes - 1:
                temp_hist.append(ns[1])
                cost_hist.append(info["energy_cost"])

            s = ns

    return Q, temp_hist, cost_hist


# ===============================
# 4) Renkler
# ===============================

def action_to_color(action):
    mapping = {
        0: 0,  # kapalı
        1: 1,  # ışık
        2: 2,  # fan
        3: 3,  # ışık+fan
        4: 2,  # fan yüksek
        5: 3,  # ışık+fan yüksek
    }
    return mapping[action]


cmap = ListedColormap([
    "#555555",  # 0 gri
    "#FFD700",  # 1 sarı
    "#000080",  # 2 lacivert
    "#1E90FF",  # 3 mavi
])


# ===============================
# 5) Fabrika GIF (4 odalı)
# ===============================

def create_factory_gif(Q_rooms, hot_flags, filename="factory.gif", frames=50):
    # Her oda için kendi ortamını başlat (aynı hot_room parametresi ile)
    envs = [FactoryRoomEnv(hot_room=hot_flags[i]) for i in range(4)]
    for e in envs:
        e.reset()

    imgs = []

    for t in range(frames):
        grid = np.zeros((6, 6), dtype=int)

        # 4 odanın her biri için greedy aksiyon seç
        for i, (r, c) in enumerate(ROOM_POSITIONS):
            state = envs[i].state
            a = int(np.argmax(Q_rooms[i][state_to_index(state)]))
            grid[r, c] = action_to_color(a)
            envs[i].step(a)

        # Çizim
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=3)
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_title("Fabrika - 4 Odalı Kontrol")
        plt.tight_layout()

        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())
        imgs.append(img)
        plt.close()

    imageio.mimsave(filename, imgs, fps=4)
    print("GIF kaydedildi:", filename)


# ===============================
# 6) PNG çıktı
# ===============================

def save_graphs(temp, cost,
                temp_file="temperature.png",
                energy_file="energy.png"):
    plt.figure()
    plt.plot(temp, marker="o")
    plt.yticks([0, 1, 2], ["Soğuk", "Konfor", "Sıcak"])
    plt.title("Sıcaklık (Son episode)")
    plt.xlabel("Adım")
    plt.ylabel("Sıcaklık durumu")
    plt.tight_layout()
    plt.savefig(temp_file, dpi=150)
    plt.close()

    plt.figure()
    plt.plot(cost, marker="o")
    plt.title("Enerji Maliyeti (Son episode)")
    plt.xlabel("Adım")
    plt.ylabel("Enerji maliyeti (göreli)")
    plt.tight_layout()
    plt.savefig(energy_file, dpi=150)
    plt.close()

    print(f"PNG dosyaları kaydedildi: {temp_file}, {energy_file}")


# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    Q_rooms = []
    temp_last = []
    cost_last = []

    # 4 odayı sırayla eğit: ilk iki oda "hot_room=True"
    for i in range(4):
        hot = HOT_ROOM_FLAGS[i]
        print(f"Oda {i+1} eğitiliyor... (hot_room={hot})")
        Q, temp, cost = train_room(
            n_episodes=800,   # istersen artır (örneğin 1500)
            max_steps=40,
            hot_room=hot
        )
        Q_rooms.append(Q)
        temp_last = temp
        cost_last = cost

    create_factory_gif(Q_rooms, HOT_ROOM_FLAGS, filename="factory.gif", frames=60)
    save_graphs(temp_last, cost_last)
