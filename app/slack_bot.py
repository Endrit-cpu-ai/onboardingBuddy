"""Slack DM bot.

    python -m app.slack_bot    # socket mode, for local dev
For a server, use http mode instead (see api.py).
"""
import logging
import time

from slack_bolt import App

from app.config import (ESCALATION_CHANNEL, OFFICE_USERGROUP, SLACK_APP_TOKEN, SLACK_BOT_TOKEN,
                        SLACK_SIGNING_SECRET)
from app.handler import handle
from app.store import log_escalation

log = logging.getLogger("slack")
bolt_app = App(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET or None)

_office = {"members": set(), "loaded": 0.0}


def in_office(client, user_id):
    """If OFFICE_USERGROUP is set, only its members get answers. Members are cached for 10 min."""
    if not OFFICE_USERGROUP:
        return True
    if time.time() - _office["loaded"] > 600:
        _office["members"] = set(client.usergroups_users_list(usergroup=OFFICE_USERGROUP)["users"])
        _office["loaded"] = time.time()
    return user_id in _office["members"]


def esc(text):
    """Escape < > & in text we didn't write, Slack treats them as markup."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def reply_blocks(reply, question):
    text = reply.text
    if reply.sources:
        text += "\n\n*Sources:* " + "  ".join(f"[{s['n']}] <{s['url']}|{esc(s['label'])}>" for s in reply.sources)
    blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": text[:2900]}}]
    # no button on escalations: they're already with a human, and on restricted ones
    # it would post the question to the channel
    if reply.outcome in ("answered", "buddy"):
        blocks.append({"type": "actions", "elements": [{
            "type": "button", "action_id": "get_human", "value": question[:1900],
            "text": {"type": "plain_text", "text": "Get a human"},
        }]})
    return blocks


def notify_team(client, user_id, question, reason, restricted=False):
    if not ESCALATION_CHANNEL:
        return
    if restricted:
        text = f":lock: <@{user_id}> has a question for HR. Please DM them. (Question text withheld on purpose.)"
    else:
        text = f":raising_hand: <@{user_id}> asked:\n>{esc(question)}\n*Why it came to you:* {reason}"
    client.chat_postMessage(channel=ESCALATION_CHANNEL, text=text)


@bolt_app.event("message")
def on_dm(event, client):
    if event.get("channel_type") != "im" or event.get("bot_id") or event.get("subtype"):
        return                                               # plain DMs from people only
    user_id, channel, question = event["user"], event["channel"], event.get("text", "").strip()
    if not question:
        return
    if not in_office(client, user_id):
        client.chat_postMessage(channel=channel, text="Onboard Buddy is only available to our office for now.")
        return

    thinking = client.chat_postMessage(channel=channel, text=":mag: Looking that up...")
    try:
        reply = handle(question, user_key=user_id)
    except Exception:
        log.exception("handle() failed")
        client.chat_update(channel=channel, ts=thinking["ts"],
                           text="Something went wrong on my side. Please try again, or ask in #onboarding-help.")
        return

    client.chat_update(channel=channel, ts=thinking["ts"], text=reply.text, blocks=reply_blocks(reply, question))
    if reply.outcome == "escalated":
        notify_team(client, user_id, question, reply.reason, restricted=reply.restricted)


@bolt_app.action("get_human")
def on_get_human(ack, body, client):
    ack()
    user_id, question = body["user"]["id"], body["actions"][0]["value"]
    log_escalation(user_id, question, "user_requested")
    notify_team(client, user_id, question, "they pressed 'Get a human'")
    client.chat_postMessage(channel=body["channel"]["id"], text="Done. Someone from the team will message you here.")


if __name__ == "__main__":
    from slack_bolt.adapter.socket_mode import SocketModeHandler

    from app.embeddings import embed_query

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s | %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    embed_query("warm up")
    print("Onboard Buddy is running (Socket Mode). DM it in Slack. Ctrl+C to stop.")
    SocketModeHandler(bolt_app, SLACK_APP_TOKEN).start()