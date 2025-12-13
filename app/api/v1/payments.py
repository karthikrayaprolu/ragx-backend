from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import stripe
import os
from app.core.config import settings

router = APIRouter(prefix="/payments", tags=["payments"])

# Initialize Stripe with the secret key from settings/env
# We assume settings has STRIPE_SECRET_KEY or we use os.getenv as fallback
stripe.api_key = os.getenv("STRIPE_SECRET_KEY", settings.STRIPE_SECRET_KEY if hasattr(settings, "STRIPE_SECRET_KEY") else "")

class CheckoutSessionRequest(BaseModel):
    priceId: str
    successUrl: str = "http://localhost:3000/dashboard?success=true"
    cancelUrl: str = "http://localhost:3000/pricing?canceled=true"

@router.post("/create-checkout-session")
async def create_checkout_session(request: CheckoutSessionRequest):
    if not stripe.api_key:
        raise HTTPException(status_code=500, detail="Stripe API key not configured")
    
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
            success_url=request.successUrl,
            cancel_url=request.cancelUrl,
        )
        return {"url": checkout_session.url}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
