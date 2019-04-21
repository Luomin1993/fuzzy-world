import pyglet
from pyglet.window import key
import ratcave as rc

# print rc.resources.obj_primitives
# Create Window and Add Keyboard State Handler to it's Event Loop
window = pyglet.window.Window()
keys = key.KeyStateHandler()
window.push_handlers(keys)

# Insert filename into WavefrontReader.
obj_filename = rc.resources.obj_primitives
obj_reader = rc.WavefrontReader(obj_filename)

# Read The Mesh Myself
# my_obj = ['obj/bed_1/bedWithTexture.obj','obj/piano_1/pianoAQueue.obj','obj/sofa_1/singleChair.obj'];
# for obj_file in my_obj:
#     my_obj_list.append( rc.WavefrontReader(obj_filename).get_mesh(???) );
#roof = rc.WavefrontReader('obj/box/box.obj').get_mesh("roof",position=(0, -3, -1.5), scale=.05);
#wall = rc.WavefrontReader('obj/box/box.obj').get_mesh("wall",position=(0, -3, -1.5), scale=.05);
#roof.textures.append(rc.Texture().from_image('obj/box/roof.jpg'))
#wall.textures.append(rc.Texture().from_image('obj/box/wall.jpg'))
box    = rc.WavefrontReader('obj/box/box.obj').get_mesh("box",position=(0, -.3, -1.5), scale=.03, rotation=(0, -90, 0));
box.uniforms['diffuse'] = 0, 0, 1
#cat    = rc.WavefrontReader('obj/box/cat.obj').get_mesh("Cat_Cube.001",position=(0, -.3, -1.5), scale=.3, rotation=(0, -90, 0));
#basket = rc.WavefrontReader('obj/box/basket.obj').get_mesh("basket",position=(0, -.3, -1.5), scale=.03, rotation=(0, -90, 0));

#box.textures.append(rc.Texture(width=333, height=333).from_image('obj/box/wall.jpg'))

# Create Mesh
monkey = obj_reader.get_mesh("Monkey", position=(0, -.2, -1.5), scale=.2)
torus = obj_reader.get_mesh("Torus", position=(-0.5, -.2, -1.5), scale=.2)
torus.uniforms['diffuse'] = 1, 1, 0
#plane = obj_reader.get_mesh("Plane", position=(0, -1, -1.5),scale = 2,rotation=(-90, 0, 0));

# Create Scene
scene = rc.Scene(meshes=[monkey, torus, box],camera=rc.Camera(orientation0=(0, 0, -1),rotation=(-20, 0, 0)))
scene.bgColor = 200, 200, 200

pass_t = 0;passed=8;
# Functions to Run in Event Loop
def rotate_meshes(dt):
    # dt is the time between frames
    global pass_t;global passed;
    if pass_t == 0:print('-----Teacher----'); print('\033[1;33;44m If the torus is stopped then the monkey can move \033[0m');
    if pass_t<passed:torus.rotation.x += 12 * dt
    if pass_t-passed>0 and pass_t-passed<0.07:print('----Agent----'); print('\033[1;32;43m Stop(torus) \033[0m');
    if pass_t>passed:monkey.position.z += .5 * dt
    pass_t+=dt;
    #print pass_t;
pyglet.clock.schedule(rotate_meshes)


def move_camera(dt):
    camera_speed = 3
    if keys[key.LEFT]:
        scene.camera.position.x -= camera_speed * dt
    if keys[key.RIGHT]:
        scene.camera.position.x += camera_speed * dt
    if keys[key.UP]:
        scene.camera.position.z += camera_speed * dt
    if keys[key.DOWN]:
        scene.camera.position.z -= camera_speed * dt
    if keys[key.K]:
        scene.camera.position.y += camera_speed * dt
    if keys[key.L]:
        scene.camera.position.y -= camera_speed * dt    
    if keys[key.H]:
        scene.camera.rotation.x += camera_speed * dt
    if keys[key.J]:
        scene.camera.rotation.x -= camera_speed * dt        
pyglet.clock.schedule(move_camera)


@window.event
def on_draw():
    with rc.default_shader:
        scene.draw()


pyglet.app.run()
