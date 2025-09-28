from jinja2 import Environment, FileSystemLoader
import copy

class JinjaConfigRenderer:
    """
    使用 Jinja2 从模板渲染配置文件的通用类。
    """

    def __init__(self,):
        self.template_paths = [
            "./templates",
        ]
        self.template_vars = {
            "max_open_trades": 1,
            "dry_run": True,
            "httpsProxy": "http://127.0.0.1:7890",
            "api_server_listen_addr": "0.0.0.0",
            "api_server_listen_port": 8080,
            "api_server_jwt_key": "3e53e2c2fad1f49dfb2b7218c0df86f271c453fa1e5f1dca4f082bd26c032c9d",
            "api_server_ws_token": "7b1JHWFUqL9atD9SkTYMsrEYu8xPkSNOww",
            "cors_origins": [
                "http://127.0.0.1:8090",
                "http://127.0.0.1:8091",
                "http://127.0.0.1:8092"
            ],
            "api_server_username": "freqtrader",
            "api_server_password": "123456",
            "initial_state": "running",
        }

        self.env = Environment(
            loader=FileSystemLoader(self.template_paths),
            trim_blocks=True,
            lstrip_blocks=True
        )
    def render_binance_config(self):
        template_file = "config_binance.json.j2"
        output_file = "config_binance.json"

        template_vars=copy.deepcopy(self.template_vars)
        template_vars["api_server_listen_port"]=8090

        template = self.env.get_template(template_file)
        rendered_config = template.render(**template_vars)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(rendered_config)
        print(f"配置文件已生成并保存到: {output_file}")


    def render_bybit_config(self):
        template_file = "config_bybit.json.j2"
        output_file = "config_bybit.json"

        template_vars=copy.deepcopy(self.template_vars)
        template_vars["api_server_listen_port"]=8091

        template = self.env.get_template(template_file)
        rendered_config = template.render(**template_vars)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(rendered_config)
        print(f"配置文件已生成并保存到: {output_file}")

    def render_okx_config(self):
        template_file = "config_okx.json.j2"
        output_file = "config_okx.json"

        template_vars=copy.deepcopy(self.template_vars)
        template_vars["api_server_listen_port"]=8092

        template = self.env.get_template(template_file)
        rendered_config = template.render(**template_vars)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(rendered_config)
        print(f"配置文件已生成并保存到: {output_file}")


    def generate_config_files(self):
        self.render_binance_config()
        self.render_bybit_config()
        self.render_okx_config()


if __name__ == "__main__":
    config_model=JinjaConfigRenderer()
    config_model.generate_config_files()