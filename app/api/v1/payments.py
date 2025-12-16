from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
import stripe
import os
from app.core.config import settings
from typing import Optional
from app.db.mongo import get_database
from app.api.v1.auth import get_current_user_id
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/payments", tags=["payments"])

# Initialize Stripe with the secret key from settings/env
# We assume settings has STRIPE_SECRET_KEY or we use os.getenv as fallback
stripe.api_key = os.getenv("STRIPE_SECRET_KEY", settings.STRIPE_SECRET_KEY if hasattr(settings, "STRIPE_SECRET_KEY") else "")

class CheckoutSessionRequest(BaseModel):
    priceId: str
    successUrl: Optional[str] = None
    cancelUrl: Optional[str] = None

@router.post("/create-checkout-session")
async def create_checkout_session(
    request: CheckoutSessionRequest,
    user_id: str = Depends(get_current_user_id)
):
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
            mode='subscription',
            success_url=success_url,
            cancel_url=cancel_url,
            client_reference_id=user_id,  # Store user_id to identify them in webhook
            metadata={'user_id': user_id}  # Additional metadata
        )
        return {"url": checkout_session.url}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/webhook", include_in_schema=False)
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events - no authentication required"""
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET", getattr(settings, "STRIPE_WEBHOOK_SECRET", ""))
    
    try:
        if webhook_secret and sig_header:
            event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
        else:
            import json
            event = json.loads(payload)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    event_type = event['type']
    
    try:
        if event_type == 'checkout.session.completed':
            session = event['data']['object']
            await handle_checkout_session_completed(session)
        
        elif event_type == 'customer.subscription.updated':
            subscription = event['data']['object']
            await handle_subscription_updated(subscription)
        
        elif event_type == 'customer.subscription.deleted':
            subscription = event['data']['object']
            await handle_subscription_deleted(subscription)
        
        elif event_type == 'invoice.payment_failed':
            invoice = event['data']['object']
            await handle_payment_failed(invoice)
            
        elif event_type in ['invoice.payment_succeeded', 'invoice.paid']:
            invoice = event['data']['object']
            await handle_invoice_payment_succeeded(invoice)
        
        elif event_type == 'customer.subscription.created':
            subscription = event['data']['object']
            await handle_subscription_created(subscription)

    except Exception as e:
        return {"status": "error", "message": str(e)}
    
    return {"status": "success"}


async def handle_subscription_created(subscription):
    """Handle subscription creation - this comes before checkout completes"""
    customer_id = subscription.get('customer')
    subscription_id = subscription.get('id')
    
    if not customer_id:
        return
    
    PRICE_ID_TO_PLAN = {
        'price_1SdltnRu2lPW20DirecI5Ata': 'starter',
        'price_1Sdlu6Ru2lPW20DiERsErBf5': 'pro',
    }
    
    plan = "pro"
    if subscription.get('items') and subscription['items'].get('data'):
        price_id = subscription['items']['data'][0]['price']['id']
        plan = PRICE_ID_TO_PLAN.get(price_id, "pro")
    
    db = await get_database()
    
    # Update by customer_id since we might not have user_id yet
    await db.users.update_one(
        {"stripe_customer_id": customer_id},
        {
            "$set": {
                "stripe_subscription_id": subscription_id,
                "subscription_status": "active",
                "plan": plan
            }
        }
    )


async def handle_checkout_session_completed(session):
    """Handle successful checkout session - links customer to user"""
    user_id = session.get('client_reference_id') or session.get('metadata', {}).get('user_id')
    customer_id = session.get('customer')
    subscription_id = session.get('subscription')
    
    if not user_id or not customer_id:
        return
    
    PRICE_ID_TO_PLAN = {
        'price_1SdltnRu2lPW20DirecI5Ata': 'starter',
        'price_1Sdlu6Ru2lPW20DiERsErBf5': 'pro',
    }
    
    plan = "pro"
    if subscription_id:
        try:
            subscription = stripe.Subscription.retrieve(subscription_id)
            if subscription.items.data:
                price_id = subscription.items.data[0].price.id
                plan = PRICE_ID_TO_PLAN.get(price_id, "pro")
        except Exception:
            pass
    
    db = await get_database()
    
    # Update or create user with both user_id and customer_id
    await db.users.update_one(
        {"user_id": user_id},
        {
            "$set": {
                "stripe_customer_id": customer_id,
                "stripe_subscription_id": subscription_id,
                "subscription_status": "active",
                "plan": plan
            }
        },
        upsert=True
    )


async def handle_subscription_updated(subscription):
    """Handle subscription updates."""
    customer_id = subscription.get('customer')
    status = subscription['status']
    
    db = await get_database()
    
    update_data = {
        "subscription_status": status,
        "updated_at": subscription['created']
    }
    
    # Also update plan if changed
    PRICE_ID_TO_PLAN = {
        'price_1SdltnRu2lPW20DirecI5Ata': 'starter',
        'price_1Sdlu6Ru2lPW20DiERsErBf5': 'pro',
    }
    
    if subscription.get('items') and subscription['items'].get('data'):
        price_id = subscription['items']['data'][0]['price']['id']
        plan = PRICE_ID_TO_PLAN.get(price_id)
        if plan:
            update_data["plan"] = plan

    await db.users.update_one(
        {"stripe_customer_id": customer_id},
        {"$set": update_data}
    )


async def handle_invoice_payment_succeeded(invoice):
    """Handle successful invoice payment (renewal)."""
    customer_id = invoice.get('customer')
    subscription_id = invoice.get('subscription')
    
    if not customer_id:
        return

    db = await get_database()
    
    # Check if we need to update plan
    update_data = {
        "subscription_status": "active",
        "updated_at": invoice['created']
    }
    
    # Try to extract plan from lines if available
    PRICE_ID_TO_PLAN = {
        'price_1SdltnRu2lPW20DirecI5Ata': 'starter',
        'price_1Sdlu6Ru2lPW20DiERsErBf5': 'pro',
    }
    
    try:
        if invoice.get('lines') and invoice['lines'].get('data'):
            for line in invoice['lines']['data']:
                if line.get('type') == 'subscription' and line.get('price'):
                    price_id = line['price']['id']
                    plan = PRICE_ID_TO_PLAN.get(price_id)
                    if plan:
                        update_data["plan"] = plan
                        break
    except Exception as e:
        pass

    await db.users.update_one(
        {"stripe_customer_id": customer_id},
        {"$set": update_data}
    )


async def handle_subscription_deleted(subscription):
    """Handle subscription cancellation."""
    customer_id = subscription.get('customer')
    
    db = await get_database()
    
    await db.users.update_one(
        {"stripe_customer_id": customer_id},
        {"$set": {
            "subscription_status": "canceled",
            "plan": "free"
        }}
    )


async def handle_payment_failed(invoice):
    """Handle failed payment."""
    customer_id = invoice.get('customer')
    
    db = await get_database()
    
    await db.users.update_one(
        {"stripe_customer_id": customer_id},
        {"$set": {
            "subscription_status": "past_due"
        }}
    )

