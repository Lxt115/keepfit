import os, json, sqlite3, datetime as dt
from contextlib import closing
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI

DB = os.getenv("DB_PATH", "data.db")
TOKEN = os.getenv("APP_TOKEN", "")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")  # V4.1-Flash, 支持图片
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY", ""), base_url="https://api.deepseek.com")


def db():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


with closing(db()) as c:
    c.executescript("""
    CREATE TABLE IF NOT EXISTS profile(id INTEGER PRIMARY KEY CHECK(id=1), sex TEXT, age INT,
      height_cm REAL, weight_kg REAL, target_weight_kg REAL, target_date TEXT);
    CREATE TABLE IF NOT EXISTS meals(id INTEGER PRIMARY KEY, date TEXT, time TEXT, meal_type TEXT,
      name TEXT, amount TEXT, kcal REAL);
    CREATE TABLE IF NOT EXISTS workouts(id INTEGER PRIMARY KEY, date TEXT, name TEXT, detail TEXT, kcal REAL);
    CREATE TABLE IF NOT EXISTS weights(id INTEGER PRIMARY KEY, date TEXT, kg REAL);
    CREATE TABLE IF NOT EXISTS sleeps(id INTEGER PRIMARY KEY, date TEXT, sleep_time TEXT, wake_time TEXT, note TEXT);
    CREATE INDEX IF NOT EXISTS i_wt ON weights(date); CREATE INDEX IF NOT EXISTS i_sl ON sleeps(date);
    CREATE INDEX IF NOT EXISTS i_m ON meals(date); CREATE INDEX IF NOT EXISTS i_w ON workouts(date);
    """)


def auth(x_token: str = Header("")):
    if TOKEN and x_token != TOKEN:
        raise HTTPException(401, "bad token")


app = FastAPI(dependencies=[Depends(auth)])


def get_profile():
    with closing(db()) as c:
        r = c.execute("SELECT * FROM profile WHERE id=1").fetchone()
    return dict(r) if r else None


class Profile(BaseModel):
    sex: str = "male"
    age: int
    height_cm: float
    weight_kg: float
    target_weight_kg: float
    target_date: str


@app.get("/api/profile")
def profile_get():
    return get_profile() or {}


@app.post("/api/profile")
def profile_set(p: Profile):
    try:
        ok = dt.date.fromisoformat(p.target_date) >= dt.date.today()
    except ValueError:
        ok = False
    if not ok:
        raise HTTPException(400, "目标日期不能早于今天")
    with closing(db()) as c:
        c.execute("REPLACE INTO profile VALUES(1,?,?,?,?,?,?)",
                  (p.sex, p.age, p.height_cm, p.weight_kg, p.target_weight_kg, p.target_date))
        c.commit()
    return {"ok": True}


def budget(burn):
    """基础代谢*1.2(久坐系数,运动单独记) - 为达成目标每日需要的热量缺口 + 运动消耗"""
    p = get_profile()
    if not p:
        return {}
    with closing(db()) as c:
        r = c.execute("SELECT kg FROM weights ORDER BY date DESC, id DESC LIMIT 1").fetchone()
    weight = r["kg"] if r else p["weight_kg"]  # 优先用最近一次称重
    bmr = 10 * weight + 6.25 * p["height_cm"] - 5 * p["age"] + (5 if p["sex"] == "male" else -161)
    base = bmr * 1.2
    try:
        days = max((dt.date.fromisoformat(p["target_date"]) - dt.date.today()).days, 1)
    except Exception:
        days = 90
    deficit = min(max((weight - p["target_weight_kg"]) * 7700 / days, 0), 1000)
    return {"bmr": round(bmr), "base": round(base), "deficit": round(deficit),
            "budget": round(base - deficit + burn)}


@app.get("/api/day")
def day(date: str):
    with closing(db()) as c:
        meals = [dict(r) for r in c.execute("SELECT * FROM meals WHERE date=? ORDER BY time,id", (date,))]
        works = [dict(r) for r in c.execute("SELECT * FROM workouts WHERE date=? ORDER BY id", (date,))]
        weights = [dict(r) for r in c.execute("SELECT * FROM weights WHERE date=? ORDER BY id", (date,))]
        sleeps = [dict(r) for r in c.execute("SELECT * FROM sleeps WHERE date=? ORDER BY id", (date,))]
    intake = sum(m["kcal"] or 0 for m in meals)
    burn = sum(w["kcal"] or 0 for w in works)
    return {"meals": meals, "workouts": works, "weights": weights, "sleeps": sleeps, "intake": round(intake), "burn": round(burn), **budget(burn)}


def sleep_minutes(sleep_time, wake_time):
    """计算睡眠时长（分钟），跨零点自动加一天；缺任一时间返回 0"""
    if not sleep_time or not wake_time:
        return 0
    try:
        a = dt.datetime.strptime(sleep_time, "%H:%M")
        b = dt.datetime.strptime(wake_time, "%H:%M")
    except ValueError:
        return 0
    m = int((b - a).total_seconds() // 60)
    return m + 1440 if m < 0 else m


@app.get("/api/month")
def month(month: str):  # YYYY-MM
    like = month + "%"
    with closing(db()) as c:
        rows = c.execute("""SELECT date, SUM(i) i, SUM(b) b FROM (
            SELECT date, kcal i, 0 b FROM meals WHERE date LIKE ? UNION ALL
            SELECT date, 0, kcal FROM workouts WHERE date LIKE ?) GROUP BY date""", (like, like)).fetchall()
        ws = c.execute("SELECT date, kg FROM weights WHERE date LIKE ? ORDER BY id", (like,)).fetchall()
        ss = c.execute("SELECT date, sleep_time, wake_time FROM sleeps WHERE date LIKE ? ORDER BY id", (like,)).fetchall()
    out = {r["date"]: {"intake": round(r["i"] or 0), "burn": round(r["b"] or 0), "weight": None, "sleep": None} for r in rows}
    for w in ws:  # 同一天多次称重取最后一次
        out.setdefault(w["date"], {"intake": 0, "burn": 0, "weight": None, "sleep": None})["weight"] = w["kg"]
    for s in ss:  # 同一天多条睡眠累加时长（分钟）
        m = sleep_minutes(s["sleep_time"], s["wake_time"])
        if m:
            d = out.setdefault(s["date"], {"intake": 0, "burn": 0, "weight": None, "sleep": None})
            d["sleep"] = (d["sleep"] or 0) + m
    for v in out.values():
        v.setdefault("weight", None)
        v.setdefault("sleep", None)
    return out


@app.delete("/api/{kind}/{id}")
def delete(kind: str, id: int):
    if kind not in ("meals", "workouts", "weights", "sleeps"):
        raise HTTPException(404)
    with closing(db()) as c:
        c.execute(f"DELETE FROM {kind} WHERE id=?", (id,))
        c.commit()
    return {"ok": True}


SYSTEM = """你是饮食与运动记录助手。把用户消息拆分成结构化记录，只输出一个JSON对象，不要任何其他文字：
{{"date":"YYYY-MM-DD","meals":[{{"time":"HH:MM","meal_type":"早餐|午餐|晚餐|加餐","name":"","amount":"","kcal":0}}],
"workouts":[{{"name":"","detail":"","kcal":0}}],"weights":[{{"kg":0.0}}],"sleeps":[{{"sleep_time":"HH:MM","wake_time":"HH:MM","note":""}}],"reply":"一句简短中文说明/估算依据，需要追问时写在这里"}}
规则：
1. 今天是{today}，现在时间{now}，用户当前查看的日期是{date}。date默认取查看日期；消息里若有明确日期（如“1001”表示10月1日，年份取今年）则用该日期。
2. 用户直接给出kcal就原样使用；否则根据食物和份量估算（拳头大小≈一个中等水果/约150-200g主食），kcal为整数。
3. 时间：用户说“刚起床/刚吃/现在”等用现在时间；说了具体时间用该时间；否则早餐08:00、午餐12:30、晚餐18:30、加餐16:00。meal_type按用户描述或时间判断。
4. 运动：每个动作/项目单独一条，name为项目名，detail保留用户原始数据（时长/组数x次数/重量），kcal按体重{weight}kg估算该项目消耗；力量训练考虑组数、次数、重量和间歇；辅助引体向上的重量是辅助重量。
5. 用户发的图片如果是食物就识别并估算；如果是营养成分表就按表计算。
6. 体重：用户提到称重/体重数值时记入weights（kg，一天可多次），不要估算体重。
7. 作息：入睡/起床时间记入sleeps，每条sleeps代表一段睡眠。“昨晚11点半睡的”=sleep_time为23:30；“刚起床”=wake_time取现在时间；只知其一时另一个填空字符串。午休单独记一条（如“中午睡了半小时”=sleep_time为13:00、wake_time为13:30，按用户描述的时间或默认13:00-13:30）。note只写用户提到的睡眠情况。作息不影响热量计算；同一条消息可同时包含饮食、运动、体重、作息。
8. 无法判断时所有数组留空，并在reply里提问。
9. 引用历史：用户说“和昨天/前天/某天吃的一样”“照旧”“同上”等时，从下方【历史饮食记录】中找到对应日期的饮食，原样复制其name、amount、kcal（meal_type、time可沿用原记录），date用用户当前要记录的日期（默认查看日期）。历史里没有对应日期时不要编造，在reply里说明并追问。
【历史饮食记录】
{history}"""


class ParseIn(BaseModel):
    text: str = ""
    images: list[str] = []  # data URI
    date: str
    today: str
    now: str


def meal_history(days: int = 7) -> str:
    """最近若干天的饮食记录，供模型处理“和昨天吃的一样”等引用"""
    with closing(db()) as c:
        rows = c.execute("SELECT date,time,meal_type,name,amount,kcal FROM meals ORDER BY date DESC,time,id").fetchall()
    by_date = {}
    for r in rows:
        by_date.setdefault(r["date"], []).append(r)
    lines = []
    for d in sorted(by_date, reverse=True)[:days]:
        items = "；".join(f'{r["meal_type"]} {r["name"]} {r["amount"] or ""} {round(r["kcal"] or 0)}kcal'.strip()
                          for r in by_date[d])
        lines.append(f"{d}: {items}")
    return "\n".join(lines) if lines else "（暂无历史记录）"


@app.post("/api/parse")
def parse(b: ParseIn):
    p = get_profile() or {}
    sys = SYSTEM.format(today=b.today, now=b.now, date=b.date, weight=p.get("weight_kg", 65),
                        history=meal_history())
    content = [{"type": "text", "text": b.text or "（见图片）"}]
    content += [{"type": "image_url", "image_url": {"url": u}} for u in b.images]
    try:
        r = client.chat.completions.create(
            model=MODEL, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": content}])
        data = json.loads(r.choices[0].message.content)
    except Exception as e:
        raise HTTPException(502, f"模型调用/解析失败: {e}")
    data.setdefault("date", b.date)
    data.setdefault("meals", [])
    data.setdefault("workouts", [])
    data.setdefault("weights", [])
    data.setdefault("sleeps", [])
    return data


class Commit(BaseModel):
    date: str
    meals: list[dict] = []
    workouts: list[dict] = []
    weights: list[dict] = []
    sleeps: list[dict] = []


@app.post("/api/commit")
def commit(b: Commit):
    with closing(db()) as c:
        for m in b.meals:
            c.execute("INSERT INTO meals(date,time,meal_type,name,amount,kcal) VALUES(?,?,?,?,?,?)",
                      (b.date, m.get("time"), m.get("meal_type"), m.get("name"), m.get("amount"), m.get("kcal", 0)))
        for w in b.workouts:
            c.execute("INSERT INTO workouts(date,name,detail,kcal) VALUES(?,?,?,?)",
                      (b.date, w.get("name"), w.get("detail"), w.get("kcal", 0)))
        for w in b.weights:
            if w.get("kg"):
                c.execute("INSERT INTO weights(date,kg) VALUES(?,?)", (b.date, w["kg"]))
        for x in b.sleeps:
            c.execute("INSERT INTO sleeps(date,sleep_time,wake_time,note) VALUES(?,?,?,?)",
                      (b.date, x.get("sleep_time"), x.get("wake_time"), x.get("note")))
        c.commit()
    return {"ok": True}


app.mount("/", StaticFiles(directory="static", html=True))
