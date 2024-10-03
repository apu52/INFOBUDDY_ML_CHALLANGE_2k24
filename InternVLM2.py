

attribute = "width"
question = f"What is the {attribute} of the product in the image? If the {attribute} isn't specified output an empty string only. You are only allowed to use the below constants with the numeric value of the attribute and have to write the full forms of the constants {entity_unit_map}"


from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

model = 'OpenGVLab/InternVL2-4B'
image = load_image('images/41uwo4PVnuL.jpg')
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192), device="cuda")
response = pipe((question, image))
print(response.text)



