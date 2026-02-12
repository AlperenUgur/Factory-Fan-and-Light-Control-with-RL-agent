import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import imageio.v2 as imageio

import torch
import torch.nn as nn
import torch.optim as optim

# ==========================
# SABİT AYARLAR
# ==========================

ROOM_POSITIONS = [(1, 1), (1, 4), (4, 1), (4, 4)]
INSULATION = [0.2, 0.4, 0.7, 0.85]  # oda1..oda4

COMFORT_LOW = 20.0
COMFORT_HIGH = 24.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# DIŞ SICAKLIK OKUMA
# ==========================

def load_outside_temps(csv_path="outside_temperature_year.csv"):
    df = pd.read_csv(csv_path)

    col = None
    for c in df.columns:
        if c.strip().lower() == "tavg":
            col = c
            break

    if col is None:
        # ilk numeric kolonu dene
        if len(df.columns) >= 2:
            col = df.columns[1]
        else:
            raise ValueError("CSV formatı uygun değil.")

    temps = df[col].dropna().tolist()
    if len(temps) < 30:
        raise ValueError("Dış sıcaklık verisi az. outside_temperature_year.csv doğru mu?")
    return temps

def temp_to_state(temp_c):
    # 0: soğuk, 1: konfor, 2: sıcak
    if temp_c < COMFORT_LOW:
        return 0
    elif temp_c <= COMFORT_HIGH:
        return 1
    else:
        return 2

# ==========================
# ORACLE-BASED ACCURACY
# ==========================

def oracle_targets(activity, inside_temp_c):

    if activity == 0:
        return 0, 0
    else:
        light = 1
        fan = 1 if inside_temp_c > COMFORT_HIGH else 0
        return light, fan

def action_to_binary(action):
    """
    action -> (light_on, fan_on)
    fan_on: fan_level > 0 ise 1
    """
    mapping = {
        0: (0, 0),
        1: (1, 0),
        2: (0, 1),
        3: (1, 1),
        4: (0, 1),
        5: (1, 1),
    }
    return mapping[action]

def evaluate_accuracy_for_room(model, outside_temps, insulation, steps=800):
    env = FactoryRoomEnv(outside_temps, insulation=insulation)
    obs = env.reset()

    correct_light = 0
    correct_fan = 0
    correct_both = 0

    for _ in range(steps):
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            q = model(x)
            action = int(torch.argmax(q, dim=1).item())

        next_obs, reward, done, info = env.step(action)

        target_light, target_fan = oracle_targets(info["activity"], info["inside_temp_c"])
        pred_light, pred_fan = action_to_binary(action)

        if pred_light == target_light:
            correct_light += 1
        if pred_fan == target_fan:
            correct_fan += 1
        if (pred_light == target_light) and (pred_fan == target_fan):
            correct_both += 1

        obs = next_obs

    light_acc = correct_light / steps
    fan_acc = correct_fan / steps
    both_acc = correct_both / steps
    return light_acc, fan_acc, both_acc

# ==========================
# ENVIRONMENT (tek oda)
# ==========================

class FactoryRoomEnv:
    """
    DQN için state'i sayısal bir vektör olarak veriyoruz.
    İç sıcaklık gerçek °C, dış sıcaklık gerçek °C.
    """

    def __init__(self, outside_temps, insulation=0.5):
        self.outside_temps = outside_temps
        self.insulation = insulation
        self.n_actions = 6

        self.day_index = 0
        self.activity = 0
        self.hour = 0
        self.price = 0
        self.inside_temp_c = 0.0

    def reset(self):
        self.day_index = random.randint(0, len(self.outside_temps) - 1)
        self.activity = random.randint(0, 1)
        self.hour = random.randint(0, 1)
        self.price = 1 if self.hour == 0 else 0  # gündüz pahalı
        self.inside_temp_c = float(self.outside_temps[self.day_index])
        return self.get_obs()

    def get_obs(self):
        outside = float(self.outside_temps[self.day_index])

        # basit normalize: -5..45 arası gibi varsayım
        def norm_temp(x):
            return (x + 5.0) / 50.0

        obs = np.array([
            float(self.activity),
            norm_temp(self.inside_temp_c),
            float(self.hour),
            float(self.price),
            norm_temp(outside),
            float(self.insulation),
        ], dtype=np.float32)

        return obs

    def step(self, action):
        # action -> (light_on, fan_level)
        mapping = {
            0: (0, 0),
            1: (1, 0),
            2: (0, 1),
            3: (1, 1),
            4: (0, 2),
            5: (1, 2),
        }
        light_on, fan_level = mapping[action]

        outside_temp_c = float(self.outside_temps[self.day_index])

        # ===== A) sıcaklık güncelle =====
        base_leak = 0.25
        leak = base_leak * (1.0 - self.insulation)

        self.inside_temp_c = self.inside_temp_c + leak * (outside_temp_c - self.inside_temp_c)

        if fan_level == 1:
            self.inside_temp_c -= 0.6
        elif fan_level == 2:
            self.inside_temp_c -= 1.2

        if self.inside_temp_c < -10:
            self.inside_temp_c = -10
        if self.inside_temp_c > 45:
            self.inside_temp_c = 45

        temp_state = temp_to_state(self.inside_temp_c)

        # ===== B) ödül =====
        reward = 0.0

        if self.activity == 1:
            if light_on == 1:
                reward += 3.0
            if temp_state == 2 and fan_level > 0:
                reward += 3.0
            if temp_state == 0 and fan_level > 0:
                reward -= 1.0
        else:
            if light_on == 1 or fan_level > 0:
                reward -= 3.0

        if self.price == 1 and self.activity == 0 and light_on == 0 and fan_level == 0:
            reward += 2.0

        # ===== C) enerji maliyeti =====
        power = 0.0

        # gündüz ışık %50, gece %100
        light_power = 0.5 if self.hour == 0 else 1.0
        if light_on == 1:
            power += light_power

        if fan_level == 1:
            power += 1.5
        elif fan_level == 2:
            power += 3.0

        price_factor = 1.0 if self.price == 0 else 2.0
        energy_cost = power * price_factor
        reward -= energy_cost * 0.4

        # ===== D) zaman ilerlet =====
        self.day_index = (self.day_index + 1) % len(self.outside_temps)

        if random.random() < 0.1:
            self.hour = 1 - self.hour
        self.price = 1 if self.hour == 0 else 0

        if random.random() < 0.3:
            self.activity = 1 - self.activity

        next_obs = self.get_obs()

        info = {
            "energy_cost": energy_cost,
            "inside_temp_c": self.inside_temp_c,
            "outside_temp_c": outside_temp_c,
            "activity": self.activity,   # accuracy için
            "hour": self.hour,
        }

        done = False
        return next_obs, reward, done, info

# ==========================
# DQN MODEL
# ==========================

class DQN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, x):
        return self.net(x)

# ==========================
# REPLAY BUFFER
# ==========================

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, s, a, r, ns):
        data = (s, a, r, ns)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns = zip(*batch)
        return np.array(s), np.array(a), np.array(r), np.array(ns)

    def __len__(self):
        return len(self.buffer)

# ==========================
# DQN TRAIN
# ==========================

def choose_action(model, obs, epsilon, n_actions):
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)
    with torch.no_grad():
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        q = model(x)
        return int(torch.argmax(q, dim=1).item())

def train_dqn_for_room(outside_temps, insulation,
                       n_episodes=400,
                       max_steps=60,
                       batch_size=64,
                       gamma=0.99,
                       lr=1e-3,
                       target_update=200):

    env = FactoryRoomEnv(outside_temps, insulation=insulation)

    obs_dim = 6
    n_actions = env.n_actions

    policy_net = DQN(obs_dim, n_actions).to(DEVICE)
    target_net = DQN(obs_dim, n_actions).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer(capacity=30000)

    step_count = 0

    last_temp = []
    last_cost = []

    for ep in range(n_episodes):
        obs = env.reset()
        epsilon = max(0.05, 1.0 - ep / n_episodes)

        for t in range(max_steps):
            action = choose_action(policy_net, obs, epsilon, n_actions)
            next_obs, reward, done, info = env.step(action)

            buffer.push(obs, action, reward, next_obs)
            obs = next_obs

            step_count += 1

            if len(buffer) >= batch_size:
                s, a, r, ns = buffer.sample(batch_size)

                s_t = torch.tensor(s, dtype=torch.float32).to(DEVICE)
                a_t = torch.tensor(a, dtype=torch.int64).unsqueeze(1).to(DEVICE)
                r_t = torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(DEVICE)
                ns_t = torch.tensor(ns, dtype=torch.float32).to(DEVICE)

                q_sa = policy_net(s_t).gather(1, a_t)

                with torch.no_grad():
                    max_next_q = target_net(ns_t).max(1, keepdim=True)[0]
                    target = r_t + gamma * max_next_q

                loss = ((q_sa - target) ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if step_count % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if ep == n_episodes - 1:
                last_temp.append(info["inside_temp_c"])
                last_cost.append(info["energy_cost"])

        if (ep + 1) % max(1, n_episodes // 5) == 0:
            print(f"Episode {ep+1}/{n_episodes}, epsilon={epsilon:.2f}")

    return policy_net, last_temp, last_cost

# ==========================
# GÖRSELLEŞTİRME
# ==========================

def action_to_color(action):
    mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 2, 5: 3}
    return mapping[action]

cmap = ListedColormap(["#555555", "#FFD700", "#000080", "#1E90FF"])

def create_factory_gif(models, outside_temps, filename="factory.gif", frames=60):
    envs = []
    for i in range(4):
        e = FactoryRoomEnv(outside_temps, insulation=INSULATION[i])
        e.reset()
        envs.append(e)

    imgs = []

    for t in range(frames):
        grid = np.zeros((6, 6), dtype=int)

        for i, (r, c) in enumerate(ROOM_POSITIONS):
            obs = envs[i].get_obs()
            with torch.no_grad():
                x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                q = models[i](x)
                a = int(torch.argmax(q, dim=1).item())

            grid[r, c] = action_to_color(a)
            envs[i].step(a)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=3)
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_title("Fabrika - 4 Oda Işık/Fan Durumu (DQN)")
        plt.tight_layout()

        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())
        imgs.append(img)
        plt.close(fig)

    imageio.mimsave(filename, imgs, fps=4)
    print("GIF kaydedildi:", filename)

def save_graphs(temp_c_list, cost_list,
                temp_file="temperature.png",
                energy_file="energy.png"):

    steps = range(len(temp_c_list))

    plt.figure(figsize=(7, 4))
    plt.plot(steps, temp_c_list, marker="o")
    plt.axhline(COMFORT_LOW, linestyle="--")
    plt.axhline(COMFORT_HIGH, linestyle="--")
    plt.title("İç Sıcaklık (Son Epizot) - °C")
    plt.xlabel("Adım")
    plt.ylabel("İç Sıcaklık (°C)")
    plt.tight_layout()
    plt.savefig(temp_file, dpi=150)
    plt.close()
    print("Kaydedildi:", temp_file)

    plt.figure(figsize=(7, 4))
    plt.plot(steps, cost_list, marker="o")
    plt.title("Enerji Maliyeti (Son Epizot)")
    plt.xlabel("Adım")
    plt.ylabel("Enerji maliyeti (göreli)")
    plt.tight_layout()
    plt.savefig(energy_file, dpi=150)
    plt.close()
    print("Kaydedildi:", energy_file)

# ==========================
# MAIN
# ==========================

if __name__ == "__main__":
    outside_temps = load_outside_temps("outside_temperature_year.csv")

    models = []
    temp_last = []
    cost_last = []

    for i in range(4):
        print(f"\nOda {i+1} DQN eğitiliyor... (insulation={INSULATION[i]})")
        model, last_temp, last_cost = train_dqn_for_room(
            outside_temps,
            insulation=INSULATION[i],
            n_episodes=400,
            max_steps=60,
            batch_size=64,
            gamma=0.99,
            lr=1e-3,
            target_update=200
        )
        models.append(model)
        temp_last = last_temp
        cost_last = last_cost

    # çıktılar
    create_factory_gif(models, outside_temps, filename="factory.gif", frames=60)
    save_graphs(temp_last, cost_last, temp_file="temperature.png", energy_file="energy.png")

    # accuracy (oracle-based)
    print("\n=== ORACLE-BASED ACCURACY ===")
    for i in range(4):
        la, fa, ba = evaluate_accuracy_for_room(models[i], outside_temps, INSULATION[i], steps=800)
        print(f"Oda {i+1}: Light Acc={la:.2f}  Fan Acc={fa:.2f}  Overall Acc={ba:.2f}")
