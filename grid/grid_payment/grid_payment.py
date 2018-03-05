import requests
from . import websockets
from ..lib import utils


class GridPayment():
    def __init__(self):
        gridhub_config = utils.get_gridhub_config()

        self.hostname = gridhub_config['hostname']
        self.protocol = gridhub_config['protocol']

        self.client_id = "61638403d5bee9d1479b81788e5c81956d6c93af8dd078ef7d012716409bead5"
        self.host = self.protocol + self.hostname

        if 'redirect' in gridhub_config.keys():
            self.redirect = gridhub_config['redirect']
        else:
            self.redirect = f"{self.host}/callback"

    def authorize(self):
        return websockets.get_token(self.hostname, self.redirect)

    def send_ether(self, email, amount, access_token=None, refresh_token=None):
        if access_token == None:
            access_token = utils.load_coinbase()['accessToken']

        if refresh_token == None:
            refresh_token = utils.load_coinbase()['refreshToken']

        send_obj = {
            'accessToken': access_token,
            'refreshToken': refresh_token,
            'email': email,
            'amount': amount
        }

        route = "/api/v0/sendEther"

        r = requests.post(self.host + route, json=send_obj)

        if r.status_code == 402:
            send_obj['two_factor_code'] = input('Enter two factor auth code: ')

            r = requests.post(self.host + route, json=send_obj)

            print(r.status_code)
        else:
            print(r.status_code)

        return r.status_code