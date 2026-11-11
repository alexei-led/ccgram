"""Guest subpackage: people outside ALLOWED_USERS who may talk to a session.

``access`` keeps the guest list and the emergency switch, ``inbound`` forwards
a guest's text into the owner's bound session, ``framing`` labels speakers in
guest chats, ``command`` is the owner's ``/guest``. Import modules by path.
"""
