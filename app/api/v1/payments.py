from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import stripe
import os
from app.core.config import settings
from typing import Optional

router = APIRouter(prefix="/payments", tags=["payments"])

# Initialize Stripe with the secret key from settings/env
# We assume settings has STRIPE_SECRET_KEY or we use os.getenv as fallback
stripe.api_key = os.getenv("STRIPE_SECRET_KEY", settings.STRIPE_SECRET_KEY if hasattr(settings, "STRIPE_SECRET_KEY") else "")

class CheckoutSessionRequest(BaseModel):
    priceId: str
    successUrl: Optional[str] = None
    cancelUrl: Optional[str] = None

@router.post("/create-checkout-session")
async def create_checkout_session(request: CheckoutSessionRequest):
    if not stripe.api_key:
        raise HTTPException(status_code=500, detail="Stripe API key not configured")
    
    # Use provided URLs or fall back to environment-configured frontend URL
    frontend_url = settings.FRONTEND_URL
    success_url = request.successUrl or f"{frontend_url}/dashboard?success=true"
    cancel_url = request.cancelUrl or f"{frontend_url}/pricing?canceled=true"
    
    try:
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[
                {
                    'price': request.priceId,
                    'quantity': 1,
                },
            ],
            mode='subscription', # Assuming subscription based on "month" pricing
            success_url=success_url,
            cancel_url=cancel_url,
        )
        return {"url": checkout_session.url}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
