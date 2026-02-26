import secrets
import bcrypt
from datetime import datetime
from fastapi import APIRouter
from fastapi.responses import JSONResponse

import dependencies as deps
from models import SignUpRequest, LoginRequest

router = APIRouter()


@router.post("/sign-up")
def sign_up(body: SignUpRequest):
    username = body.username.strip()
    password = body.password
    confirm_password = body.confirmPassword

    if not username or len(username) < 3:
        return JSONResponse({"success": False, "message": "Username must be at least 3 characters"}, status_code=400)
    if not password or len(password) < 6:
        return JSONResponse({"success": False, "message": "Password must be at least 6 characters"}, status_code=400)
    if password != confirm_password:
        return JSONResponse({"success": False, "message": "Passwords do not match"}, status_code=400)

    existing = deps.users_collection.find_one({"username": username})
    if existing:
        return JSONResponse({"success": False, "message": "Username is already taken"}, status_code=409)

    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    user = {
        "username": username,
        "password_hash": password_hash,
        "created_at": datetime.utcnow(),
        "status": "active",
    }
    result = deps.users_collection.insert_one(user)
    token = secrets.token_urlsafe(32)

    return {
        "success": True,
        "message": "Account created successfully!",
        "data": {"id": str(result.inserted_id), "username": username, "is_guest": False},
        "token": token,
    }


@router.post("/login")
def login(body: LoginRequest):
    username = body.username.strip()
    password = body.password

    if not username or not password:
        return JSONResponse({"success": False, "message": "Username and password required"}, status_code=400)

    user = deps.users_collection.find_one({"username": username})
    if not user:
        return JSONResponse({"success": False, "message": "Invalid username or password"}, status_code=401)
    if not bcrypt.checkpw(password.encode("utf-8"), user["password_hash"]):
        return JSONResponse({"success": False, "message": "Invalid username or password"}, status_code=401)
    if user.get("status") != "active":
        return JSONResponse({"success": False, "message": "User account is not active"}, status_code=401)

    token = secrets.token_urlsafe(32)
    return {
        "success": True,
        "message": "Successfully logged in",
        "data": {"id": str(user["_id"]), "username": user["username"], "is_guest": False},
        "token": token,
    }


@router.post("/guest")
def guest():
    return {
        "success": True,
        "message": "Continuing as Guest",
        "data": {"id": None, "username": "Guest", "is_guest": True},
    }
