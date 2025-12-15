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


@router.post("/webhook")
async def stripe_webhook(request: Request):
    """
    Handle Stripe webhook events.
    This endpoint is called by Stripe when events occur (e.g., successful payment).
    """
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET", settings.STRIPE_WEBHOOK_SECRET if hasattr(settings, "STRIPE_WEBHOOK_SECRET") else "")
    
    if not webhook_secret:
        logger.warning("Stripe webhook secret not configured")
        # In development, you might want to process events anyway
        # In production, you should require the webhook secret
        if not settings.TEST_MODE:
            raise HTTPException(status_code=500, detail="Webhook secret not configured")
    
    try:
        if webhook_secret and sig_header:
            event = stripe.Webhook.construct_event(
                payload, sig_header, webhook_secret
            )
        else:
            # For testing without webhook secret
            import json
            event = json.loads(payload)
    except ValueError as e:
        logger.error(f"Invalid payload: {e}")
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Invalid signature: {e}")
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    # Handle the event
    event_type = event['type']
    logger.info(f"Received Stripe event: {event_type}")
    
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
    
    except Exception as e:
        logger.error(f"Error processing webhook event {event_type}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    return {"status": "success"}


async def handle_checkout_session_completed(session):
    """Handle successful checkout session."""
    user_id = session.get('client_reference_id') or session.get('metadata', {}).get('user_id')
    
    if not user_id:
        logger.error("No user_id found in checkout session")
        return
    
    customer_id = session.get('customer')
    subscription_id = session.get('subscription')
    
    # Get subscription details to determine the plan
    plan = "pro"  # Default
    if subscription_id:
        try:
            subscription = stripe.Subscription.retrieve(subscription_id)
            # Get the price ID from the subscription
            if subscription.items.data:
                price_id = subscription.items.data[0].price.id
                # Map price IDs to plan names
                if 'starter' in price_id.lower():
                    plan = "starter"
                elif 'pro' in price_id.lower():
                    plan = "pro"
                elif 'business' in price_id.lower():
                    plan = "business"
        except Exception as e:
            logger.error(f"Error retrieving subscription: {e}")
    
    db = await get_database()
    
    # Update user with subscription info
    await db.users.update_one(
        {"user_id": user_id},
        {
            "$set": {
                "stripe_customer_id": customer_id,
                "stripe_subscription_id": subscription_id,
                "subscription_status": "active",
                "plan": plan,
                "updated_at": stripe.util.convert_to_stripe_object(session)['created']
            }
        },
        upsert=True
    )
    
    logger.info(f"Updated user {user_id} with {plan} plan")


async def handle_subscription_updated(subscription):
    """Handle subscription updates."""
    customer_id = subscription.get('customer')
    subscription_id = subscription['id']
    status = subscription['status']
    
    db = await get_database()
    
    # Find user by customer_id
    user = await db.users.find_one({"stripe_customer_id": customer_id})
    
    if not user:
        logger.warning(f"No user found for customer {customer_id}")
        return
    
    await db.users.update_one(
        {"stripe_customer_id": customer_id},
        {"$set": {
            "subscription_status": status,
            "updated_at": subscription['created']
        }}
    )
    
    logger.info(f"Updated subscription status for customer {customer_id} to {status}")


async def handle_subscription_deleted(subscription):
    """Handle subscription cancellation."""
    customer_id = subscription.get('customer')
    
    db = await get_database()
    
    await db.users.update_one(
        {"stripe_customer_id": customer_id},
        {"$set": {
            "subscription_status": "canceled",
            "plan": "free",  # Downgrade to free
            "updated_at": subscription['created']
        }}
    )
    
    logger.info(f"Subscription canceled for customer {customer_id}")


async def handle_payment_failed(invoice):
    """Handle failed payment."""
    customer_id = invoice.get('customer')
    subscription_id = invoice.get('subscription')
    
    db = await get_database()
    
    await db.users.update_one(
        {"stripe_customer_id": customer_id},
        {"$set": {
            "subscription_status": "past_due",
            "updated_at": invoice['created']
        }}
    )
    
    logger.warning(f"Payment failed for customer {customer_id}")
