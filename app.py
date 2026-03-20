from flask import Flask, render_template, Response, request, redirect, session, url_for, jsonify
import cv2
import numpy as np
import torch
import random
import smtplib
import sqlite3
import os
import time
from email.message import EmailMessage
from stable_baselines3 import DQN

from drone_env import DroneEnv, Drone, WIDTH, HEIGHT

app = Flask(__name__)
app.secret_key = "drone_secret_key"

@app.route('/favicon.ico')
def favicon():
    return app.send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )

# ---------------------------
# DATABASE SETUP
# ---------------------------
def init_db():
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()

# ---------------------------
# EMAIL SETTINGS
# ---------------------------
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "prosdgunal@gmail.com").strip()
# Gmail app passwords are often copied with spaces; strip them for SMTP login.
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "jkmk qokw sqle mqpg").replace(" ", "")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "darshinihema2102@gmail.com").strip()
ALERT_COOLDOWN_SECONDS = 60
last_alert_sent_at = {}

# ---------------------------
# DRONE SETTINGS
# ---------------------------
NUM_DRONES = 3
NUM_OBSTACLES = 3

OBSTACLE_W = 80
OBSTACLE_H = 80

# ---------------------------
# RANDOM BUILDINGS
# ---------------------------
obstacles = [
    (
        random.randint(50, WIDTH - OBSTACLE_W - 50),
        random.randint(50, HEIGHT - OBSTACLE_H - 50),
        OBSTACLE_W,
        OBSTACLE_H
    )
    for _ in range(NUM_OBSTACLES)
]

# ---------------------------
# RL ENVIRONMENT
# ---------------------------
env = DroneEnv(WIDTH, HEIGHT, obstacles)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = DQN("MlpPolicy", env, verbose=0, device=device)

print("Training RL Model...")
model.learn(total_timesteps=5000)
print("Training Complete")

# ---------------------------
# DRONES
# ---------------------------
drones = [
    Drone(i, DroneEnv(WIDTH, HEIGHT, obstacles), model)
    for i in range(NUM_DRONES)
]

# ---------------------------
# EMAIL ALERT FUNCTION
# ---------------------------
def send_damage_alert(drone_id, x, y):

    if not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_RECEIVER:
        print("Email skipped: sender/password/receiver is missing")
        return False

    now = time.time()
    last_sent = last_alert_sent_at.get(drone_id, 0)
    if now - last_sent < ALERT_COOLDOWN_SECONDS:
        return False

    msg = EmailMessage()
    msg["Subject"] = "Drone Damage Alert"
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER

    msg.set_content(f"""
Drone Damage Detected!

Drone ID : {drone_id}
Location : ({x},{y})

Drone collided with building.
""")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)

        last_alert_sent_at[drone_id] = now
        print("Email sent")
        return True

    except Exception as e:
        print("Email error:", e)
        return False

# ---------------------------
# FRAME GENERATOR
# ---------------------------
def generate_frames():

    while True:

        frame = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255

        for obs in obstacles:
            x, y, w, h = obs
            cv2.rectangle(frame,(x,y),(x+w,y+h),(150,150,150),-1)

        for drone in drones:

            drone.move(drones)

            damage = False

            for obs in obstacles:

                ox, oy, w, h = obs

                if ox <= drone.x <= ox+w and oy <= drone.y <= oy+h:

                    damage = True
                    send_damage_alert(drone.id,int(drone.x),int(drone.y))
                    drone.reset()

            color = (0,0,255) if damage else (255,0,0)

            cv2.circle(frame,(int(drone.x),int(drone.y)),10,color,-1)

            cv2.circle(frame,(int(drone.target[0]),int(drone.target[1])),5,(0,255,0),-1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ---------------------------
# ROUTES
# ---------------------------

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/login", methods=["GET","POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("users.db")
        cur = conn.cursor()

        cur.execute("SELECT * FROM users WHERE username=? AND password=?",(username,password))
        user = cur.fetchone()

        conn.close()

        if user:
            session["user"] = username
            return redirect("/dashboard")

    return render_template("login.html")


@app.route("/register", methods=["GET","POST"])
def register():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("users.db")
        cur = conn.cursor()

        cur.execute("INSERT INTO users(username,password) VALUES(?,?)",(username,password))

        conn.commit()
        conn.close()

        return redirect("/login")

    return render_template("register.html")


@app.route("/dashboard")
def dashboard():

    if "user" not in session:
        return redirect("/login")

    return render_template("dashboard.html")


@app.route("/video")
def video():
    return Response(generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


@app.route("/test-email")
def test_email():

    if "user" not in session:
        return redirect("/login")

    sent = send_damage_alert("TEST", 0, 0)
    return "Test email sent" if sent else "Test email failed or skipped"


@app.route("/fire-alert", methods=["POST"])
def fire_alert():

    if "user" not in session:
        return jsonify({"ok": False, "error": "Unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    drone_id = data.get("drone_id", "UI")
    location_x = int(data.get("x", 0))
    location_y = int(data.get("y", 0))

    sent = send_damage_alert(drone_id, location_x, location_y)

    if sent:
        return jsonify({"ok": True, "message": "Email alert sent"})

    return jsonify({"ok": False, "message": "Email not sent (failed or cooldown active)"}), 429


# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)