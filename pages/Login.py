import hashlib
import hmac
import json
import os
import random
import smtplib
import ssl
import time
from email.mime.text import MIMEText
from pathlib import Path

import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Sign in — SolaREAL",
    page_icon="✦",
    layout="centered",
)

# ============================================================
# PATHS
# ============================================================

script_dir = Path(__file__).parent      # pages/
project_root = script_dir.parent        # solareal_app/
DATA_DIR = project_root / "data"
USERS_PATH = DATA_DIR / "users.json"
DATA_DIR.mkdir(parents=True, exist_ok=True)


DETECTOR_PAGE = "pages/1_Detector.py"

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_EMAIL = os.environ.get("SOLAREAL_SMTP_EMAIL", "solareal.izzi@gmail.com")
SMTP_PASSWORD = os.environ.get("SOLAREAL_SMTP_PASSWORD", "rggabsxkkeyozesw")
CODE_TTL_SECONDS = 15 * 60  # a verification code is valid for 15 minutes


def send_verification_email(to_email, code):
    if not SMTP_EMAIL or not SMTP_PASSWORD:
        raise RuntimeError(
            "SOLAREAL_SMTP_EMAIL / SOLAREAL_SMTP_PASSWORD are not set. "
            "See the EMAIL CONFIG comment near the top of this file."
        )
    message = MIMEText(
        f"Your SolaREAL verification code is: {code}\n\n"
        f"This code expires in {CODE_TTL_SECONDS // 60} minutes. "
        f"If you didn't request this, you can ignore this email."
    )
    message["Subject"] = "Confirm your SolaREAL account"
    message["From"] = SMTP_EMAIL
    message["To"] = to_email

    context = ssl.create_default_context()
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls(context=context)
        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        server.sendmail(SMTP_EMAIL, to_email, message.as_string())


def _generate_code():
    return f"{random.randint(0, 999999):06d}"


def _load_users():
    if USERS_PATH.exists():
        with open(USERS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_users(users):
    with open(USERS_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def _hash_password(password, salt=None):
    if salt is None:
        salt = os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), 100_000
    ).hex()
    return salt, digest


def _verify_password(password, salt, expected_digest):
    _, digest = _hash_password(password, salt)
    return hmac.compare_digest(digest, expected_digest)


def create_user(username, email, password):
    """Creates an unverified account and returns (True, code) or (False, error_message)."""
    users = _load_users()
    key = username.strip().lower()
    if not key:
        return False, "Please choose a username."
    if key in users:
        return False, "This username is already taken."
    salt, digest = _hash_password(password)
    code = _generate_code()
    users[key] = {
        "username": username.strip(),
        "email": email.strip(),
        "salt": salt,
        "password_hash": digest,
        "created_at": time.time(),
        "verified": False,
        "verification_code": code,
        "verification_code_expires": time.time() + CODE_TTL_SECONDS,
    }
    _save_users(users)
    return True, code


def issue_new_code(username):
    """Generates and stores a fresh code for an existing (unverified) user. Returns the code or None."""
    users = _load_users()
    key = username.strip().lower()
    if key not in users:
        return None
    code = _generate_code()
    users[key]["verification_code"] = code
    users[key]["verification_code_expires"] = time.time() + CODE_TTL_SECONDS
    _save_users(users)
    return code


def verify_code(username, code):
    users = _load_users()
    key = username.strip().lower()
    user = users.get(key)
    if not user:
        return False, "Account not found."
    if user.get("verified"):
        return True, user
    if time.time() > user.get("verification_code_expires", 0):
        return False, "This code has expired — request a new one below."
    if not hmac.compare_digest(str(code).strip(), str(user.get("verification_code", ""))):
        return False, "Incorrect code."
    user["verified"] = True
    user.pop("verification_code", None)
    user.pop("verification_code_expires", None)
    users[key] = user
    _save_users(users)
    return True, user


def authenticate(username, password):
    users = _load_users()
    key = username.strip().lower()
    user = users.get(key)
    if not user:
        return False, "No account found with that username."
    if not _verify_password(password, user["salt"], user["password_hash"]):
        return False, "Incorrect password."
    if not user.get("verified", False):
        return False, "UNVERIFIED"
    return True, user


# ============================================================
# SESSION STATE
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "pending_verification_user" not in st.session_state:
    st.session_state.pending_verification_user = None

if st.session_state.logged_in:
    st.success(f"You're already signed in as **{st.session_state.username}**.")
    if st.button("Go to detector", use_container_width=True):
        st.switch_page(DETECTOR_PAGE)
    st.stop()


css_path = project_root / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="topbar">
      <div class="brand"><span class="brand-mark">✦</span><span class="brand-name">SolaREAL</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.pending_verification_user:
    pending_user = st.session_state.pending_verification_user

    st.markdown('<h1 class="page-title">Check your inbox</h1>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="page-subtitle">Enter the 6-digit code we sent to confirm your account.</p>',
        unsafe_allow_html=True,
    )

    with st.form("verify_form"):
        code_input = st.text_input("Verification code", max_chars=6)
        verify_submitted = st.form_submit_button(
            "Verify email", type="primary", use_container_width=True
        )

    if verify_submitted:
        ok, result = verify_code(pending_user, code_input)
        if ok:
            st.session_state.logged_in = True
            st.session_state.username = result["username"]
            st.session_state.pending_verification_user = None
            st.switch_page(DETECTOR_PAGE)
        else:
            st.error(result)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Resend code", use_container_width=True):
            new_code = issue_new_code(pending_user)
            if new_code is None:
                st.error("Account not found.")
            else:
                try:
                    users = _load_users()
                    to_email = users[pending_user.strip().lower()]["email"]
                    send_verification_email(to_email, new_code)
                    st.success(f"New code sent to {to_email}.")
                except Exception as exc:
                    st.warning(f"Couldn't send the email ({exc}). Dev fallback — your code is: {new_code}")
    with col2:
        if st.button("Use a different account", use_container_width=True):
            st.session_state.pending_verification_user = None
            st.rerun()

    st.stop()

# ------------------------------------------------------------
# LOGIN / SIGNUP TABS
# ------------------------------------------------------------
st.markdown('<h1 class="page-title">Sign in to continue</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="page-subtitle">Create a free account or log in to analyze your video.</p>',
    unsafe_allow_html=True,
)

if st.session_state.get("pending_video_name"):
    st.info(
        f"We'll pick up right where you left off with "
        f"**{st.session_state.pending_video_name}** once you're signed in."
    )

login_tab, signup_tab = st.tabs(["Log In", "Sign Up"])

# ------------------------------------------------------------
# LOG IN
# ------------------------------------------------------------
with login_tab:
    with st.form("login_form"):
        login_username = st.text_input("Username")
        login_password = st.text_input("Password", type="password")
        login_submitted = st.form_submit_button(
            "Log In", type="primary", use_container_width=True
        )

    if login_submitted:
        if not login_username or not login_password:
            st.error("Please fill in both fields.")
        else:
            ok, result = authenticate(login_username, login_password)
            if ok:
                st.session_state.logged_in = True
                st.session_state.username = result["username"]
                st.switch_page(DETECTOR_PAGE)
            elif result == "UNVERIFIED":
                # Give them a fresh code and send them to the verify step.
                new_code = issue_new_code(login_username)
                users = _load_users()
                to_email = users[login_username.strip().lower()]["email"]
                try:
                    send_verification_email(to_email, new_code)
                    st.info(f"Your account isn't verified yet — we just sent a new code to {to_email}.")
                except Exception as exc:
                    st.warning(f"Account not verified, and the email couldn't be sent ({exc}). Dev fallback — your code is: {new_code}")
                st.session_state.pending_verification_user = login_username.strip()
                st.rerun()
            else:
                st.error(result)

# ------------------------------------------------------------
# SIGN UP
# ------------------------------------------------------------
with signup_tab:
    with st.form("signup_form"):
        signup_username = st.text_input("Choose a username")
        signup_email = st.text_input("Email")
        signup_password = st.text_input("Choose a password", type="password")
        signup_password_confirm = st.text_input("Confirm password", type="password")
        signup_submitted = st.form_submit_button(
            "Create Account", type="primary", use_container_width=True
        )

    if signup_submitted:
        if not signup_username or not signup_email or not signup_password:
            st.error("Please fill in all fields.")
        elif "@" not in signup_email or "." not in signup_email.split("@")[-1]:
            st.error("Please enter a valid email address.")
        elif signup_password != signup_password_confirm:
            st.error("Passwords do not match.")
        elif len(signup_password) < 6:
            st.error("Password must be at least 6 characters.")
        else:
            ok, result = create_user(signup_username, signup_email, signup_password)
            if ok:
                code = result
                try:
                    send_verification_email(signup_email, code)
                    st.success(f"Account created — a verification code was sent to {signup_email}.")
                except Exception as exc:
                    st.warning(f"Account created, but the email couldn't be sent ({exc}). Dev fallback — your code is: {code}")
                st.session_state.pending_verification_user = signup_username.strip()
                st.rerun()
            else:
                st.error(result)
