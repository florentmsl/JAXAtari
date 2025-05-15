import os
from functools import partial
from typing import NamedTuple, Tuple, List, Dict, Any, Optional

import jax
import jax.numpy as jnp
import chex
import pygame

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
import jaxatari.rendering.atraJaxis as aj
from jaxatari.renderers import AtraJaxisRenderer

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

class GameConfig(NamedTuple):
    screen_width: int = 160
    screen_height: int = 210

    # Player (Sir Lancelot)
    player_width: int = 9
    player_height: int = 14
    player_start_x: int = 20
    player_start_y: int = 120  # roughly mid‑screen

    # Physics
    gravity: float = 0.5        # pixels / frame² pulling downwards
    flap_impulse: float = -4.0  # vertical impulse when FIRE is pressed
    max_fall_speed: float = 3.0 # terminal velocity
    max_speed_x: int = 2        # horizontal speed in pixels/frame

    # Enemies (Flying Snakes – first level)
    enemy_width: int = 9
    enemy_height: int = 12
    enemy_speed: int = 1        # horizontal speed
    enemy_spawn_y_positions: Tuple[int, ...] = (60, 90, 120, 150)
    enemy_spawn_x_left: int = -10
    enemy_spawn_x_right: int = 170

    # Scoring / lives
    initial_lives: int = 3
    points_per_enemy: int = 250


# -----------------------------------------------------------------------------
# STATE REPRESENTATIONS
# -----------------------------------------------------------------------------

class PlayerState(NamedTuple):
    x: chex.Array
    y: chex.Array
    vel_y: chex.Array
    facing_left: chex.Array  # Boolean – determines sprite & collision outcome
    flap_cooldown: chex.Array  # simple debounce so holding FIRE doesn’t spam

class EnemyState(NamedTuple):
    positions: chex.Array  # shape (N, 2) – x, y
    directions: chex.Array  # shape (N,) – -1 == moving left → right? we’ll encode -1|1
    active: chex.Array       # shape (N,) – whether the slot is currently on‑screen
    animation_counter: chex.Array  # used to switch sprite frames

class GameState(NamedTuple):
    player: PlayerState
    enemies: EnemyState
    score: chex.Array
    lives: chex.Array
    time: chex.Array   # frame counter (0‑based)
    game_over: chex.Array

# -----------------------------------------------------------------------------
# OBSERVATION / INFO (for RL agents or debugging  – minimal for now)
# -----------------------------------------------------------------------------

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray

class SirLancelotObservation(NamedTuple):
    player: EntityPosition
    enemies: jnp.ndarray  # (N,4)
    score: jnp.ndarray
    lives: jnp.ndarray

class SirLancelotInfo(NamedTuple):
    time: jnp.ndarray

# -----------------------------------------------------------------------------
# ENVIRONMENT IMPLEMENTATION
# -----------------------------------------------------------------------------

class JaxSirLancelot(JaxEnvironment[GameState, SirLancelotObservation, SirLancelotInfo]):
    """First‑level implementation of *Sir Lancelot* (Atari 2600, 1983).

    Only the exterior castle screen with flying snakes is implemented.  The
    player rides Pegasus, flapping to stay aloft, and must defeat all snakes
    with a jousting mechanic: on collision the higher combatant wins; equal
    heights or both facing away == bounce (no kill).
    """

    def __init__(self, max_enemies: int = 4):
        super().__init__()
        self.cfg = GameConfig()
        self.max_enemies = max_enemies
        self.state = self.reset()[1]  # store so render() outside env can use

    # ------------------------------------------------------------------ RESET
    def reset(self, key: jax.random.PRNGKey = None):
        cfg = self.cfg
        player = PlayerState(
            x=jnp.array(cfg.player_start_x, dtype=jnp.float32),
            y=jnp.array(cfg.player_start_y, dtype=jnp.float32),
            vel_y=jnp.array(0.0),
            facing_left=jnp.array(False),
            flap_cooldown=jnp.array(0),
        )

        # Spawn snakes alternating left/right
        positions = []
        directions = []
        for i, y in enumerate(cfg.enemy_spawn_y_positions[: self.max_enemies]):
            if i % 2 == 0:
                positions.append([cfg.enemy_spawn_x_left, y])
                directions.append(1)  # fly right
            else:
                positions.append([cfg.enemy_spawn_x_right, y])
                directions.append(-1)  # fly left
        positions = jnp.array(positions, dtype=jnp.float32)
        directions = jnp.array(directions, dtype=jnp.int32)
        active = jnp.ones((self.max_enemies,), dtype=bool)
        animation_counter = jnp.zeros((self.max_enemies,), dtype=jnp.int32)

        enemies = EnemyState(
            positions=positions,
            directions=directions,
            active=active,
            animation_counter=animation_counter,
        )

        state = GameState(
            player=player,
            enemies=enemies,
            score=jnp.array(0),
            lives=jnp.array(cfg.initial_lives),
            time=jnp.array(0),
            game_over=jnp.array(False),
        )
        return self._get_observation(state), state

    # ------------------------------------------------------------------ STEP
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: GameState, action: int):
        cfg = self.cfg

        # --------------------------- Handle player input
        # Horizontal movement (no acceleration for simplicity)
        dx = jnp.where(
            jnp.any(jnp.array([action == Action.LEFT, action == Action.UPLEFT, action == Action.DOWNLEFT])),
            -cfg.max_speed_x,
            jnp.where(
                jnp.any(jnp.array([action == Action.RIGHT, action == Action.UPRIGHT, action == Action.DOWNRIGHT])),
                cfg.max_speed_x,
                0,
            ),
        )

        # Flap (FIRE) – impulse upward if cooldown reached zero
        fire_pressed = jnp.any(
            jnp.array([
                action == Action.FIRE,
                action == Action.UPFIRE,
                action == Action.DOWNFIRE,
                action == Action.LEFTFIRE,
                action == Action.RIGHTFIRE,
            ])
        )
        can_flap = state.player.flap_cooldown == 0
        vel_y = state.player.vel_y + cfg.gravity  # gravity each frame
        vel_y = jnp.clip(vel_y, -999, cfg.max_fall_speed)
        # apply flap impulse
        vel_y = jax.lax.cond(
            fire_pressed & can_flap,
            lambda vy: vy + cfg.flap_impulse,
            lambda vy: vy,
            vel_y,
        )
        new_cooldown = jax.lax.cond(
            fire_pressed & can_flap,
            lambda _: jnp.array(5),  # 5‑frame cooldown between flaps
            lambda _: jnp.maximum(state.player.flap_cooldown - 1, 0),
            operand=None,
        )

        new_x = jnp.clip(state.player.x + dx, 0, cfg.screen_width - cfg.player_width)
        new_y = jnp.clip(state.player.y + vel_y, 0, cfg.screen_height - cfg.player_height - 20)  # leave space for HUD

        facing_left = jax.lax.cond(dx < 0, lambda: True, lambda: jax.lax.cond(dx > 0, lambda: False, lambda: state.player.facing_left))

        player = PlayerState(
            x=new_x,
            y=new_y,
            vel_y=vel_y,
            facing_left=facing_left,
            flap_cooldown=new_cooldown,
        )

        # --------------------------- Update enemies
        def update_enemy(i, tup):
            pos, dir_, active, anim_counter = tup

            # Move only if active
            new_x = jnp.where(active, pos[0] + dir_ * cfg.enemy_speed, pos[0])
            # wrap around horizontally outside screen, keep same dir
            off_left = new_x > cfg.screen_width + 10
            off_right = new_x < -10
            still_active = jnp.where(off_left | off_right, False, active)
            # update animation counter (toggle 0/1 every 8 frames)
            new_anim = (anim_counter + 1) % 16
            return (jnp.array([new_x, pos[1]]), dir_, still_active, new_anim)

        positions, directions, actives, anims = state.enemies
        positions_out, directions_out, actives_out, anims_out = jax.lax.fori_loop(
            0, self.max_enemies, lambda i, carry: (*[x.at[i].set(y) for x, y in zip(carry, update_enemy(i, (positions[i], directions[i], actives[i], anims[i])))],), (positions, directions, actives, anims)
        )
        # The above trick isn’t nice – easier: build lists then stack.
        # Simpler: use vmap

        # --- update all enemies in one vmap pass ---
        def move_enemy(pos, dir_, active, anim):
            new_x = jnp.where(active, pos[0] + dir_ * cfg.enemy_speed, pos[0])
            off_screen = (new_x < -10) | (new_x > cfg.screen_width + 10)
            new_active = jnp.where(off_screen, False, active)
            new_anim   = (anim + 1) % 16
            return jnp.array([new_x, pos[1]]), dir_, new_active, new_anim

        positions_out, directions_out, actives_out, anims_out = jax.vmap(move_enemy)(
            state.enemies.positions,
            state.enemies.directions,
            state.enemies.active,
            state.enemies.animation_counter,
        )

        enemies = EnemyState(
            positions=positions_out,
            directions=directions_out,
            active=actives_out,
            animation_counter=anims_out,
        )

        # --------------------------- Collision detection
        def collide_single(enemy_pos, enemy_active):
            collides = enemy_active & (
                (player.x < enemy_pos[0] + cfg.enemy_width) &
                (player.x + cfg.player_width > enemy_pos[0]) &
                (player.y < enemy_pos[1] + cfg.enemy_height) &
                (player.y + cfg.player_height > enemy_pos[1])
            )
            return collides

        collision_mask = jax.vmap(collide_single)(enemies.positions, enemies.active)
        any_collision = jnp.any(collision_mask)

        # Determine outcome per collision rules
        def resolve_collision(enemy_idx, vals):
            pos = enemies.positions[enemy_idx]
            active = enemies.active[enemy_idx]
            higher_player = player.y < pos[1]  # lower `y` is higher on screen
            player_wins = higher_player  # simplified rule: higher wins
            # update active flag
            new_active = jnp.where(active & collision_mask[enemy_idx] & player_wins, False, active)
            score_add = jnp.where(active & collision_mask[enemy_idx] & player_wins, cfg.points_per_enemy, 0)
            player_loses = (~player_wins) & active & collision_mask[enemy_idx]
            life_loss = jnp.where(player_loses, 1, 0)
            return new_active, score_add, life_loss

        def loop_body(i, carry):
            actives_arr, sc, life_loss_acc = carry
            new_active_i, score_add_i, life_loss_i = resolve_collision(i, None)
            actives_arr = actives_arr.at[i].set(new_active_i)
            return actives_arr, sc + score_add_i, life_loss_acc + life_loss_i

        actives_after, score_gained, lives_lost = jax.lax.fori_loop(0, self.max_enemies, loop_body, (enemies.active, 0, 0))
        enemies = enemies._replace(active=actives_after)

        new_score = state.score + score_gained
        new_lives = state.lives - lives_lost

        # --------------------------- Check win / loss conditions
        all_enemies_defeated = jnp.all(~enemies.active)
        game_over = (new_lives <= 0) | all_enemies_defeated

        new_state = GameState(
            player=player,
            enemies=enemies,
            score=new_score,
            lives=new_lives,
            time=state.time + 1,
            game_over=game_over,
        )

        reward = new_score - state.score
        done = self._get_done(new_state)
        obs = self._get_observation(new_state)
        info = self._get_info(new_state)

        return obs, new_state, reward, done, info

    # ------------------------------------------------------------------ HELPERS
    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: GameState):
        player_ent = EntityPosition(
            x=state.player.x,
            y=state.player.y,
            width=jnp.array(self.cfg.player_width),
            height=jnp.array(self.cfg.player_height),
        )
        enemy_ents = jnp.concatenate([
            state.enemies.positions,
            jnp.full((self.max_enemies, 1), self.cfg.enemy_width),
            jnp.full((self.max_enemies, 1), self.cfg.enemy_height),
        ], axis=1)
        return SirLancelotObservation(
            player=player_ent,
            enemies=enemy_ents,
            score=state.score,
            lives=state.lives,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: GameState):
        return SirLancelotInfo(time=state.time)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: GameState):
        return state.game_over

    @partial(jax.jit, static_argnums=(0,))
    def get_action_space(self):
        return jnp.array([
            Action.NOOP,
            Action.LEFT,
            Action.RIGHT,
            Action.UP,
            Action.DOWN,
            Action.FIRE,  # flap
            Action.UPLEFT,
            Action.UPRIGHT,
            Action.DOWNLEFT,
            Action.DOWNRIGHT,
        ])

# -----------------------------------------------------------------------------
# RENDERER
# -----------------------------------------------------------------------------

class SirLancelotRenderer(AtraJaxisRenderer):
    """Draws current *Sir Lancelot* level 1 state to a 160×210 RGB raster."""

    def __init__(self):
        super().__init__()
        self.cfg = GameConfig()
        self.sprites = self._load_sprites()

    # ----------------------------------------------------------- sprite loader
    def _load_sprites(self) -> Dict[str, chex.Array]:
        MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
        sprite_path = os.path.join(MODULE_DIR, "sprites/sir_lancelot/")

        def load(name: str):
            return aj.loadFrame(os.path.join(sprite_path, name + ".npy")).astype(jnp.uint8)

        sprites: Dict[str, chex.Array] = {}
        sprites["background"] = jnp.expand_dims(load("Background_lvl1"), 0)
        # --- Player frames ----------------------------------------------------
        player_neutral = load("SirLancelot_lvl1_neutral")
        player_fly     = load("SirLancelot_lvl1_fly")
        player_frames  = aj.pad_to_match([player_neutral, player_fly])
        sprites["player_neutral"] = jnp.expand_dims(player_frames[0], 0)
        sprites["player_fly"]     = jnp.expand_dims(player_frames[1], 0)
        # Beast animation: 2 frames – we’ll repeat each for 8 ticks (v‑similar to Seaquest)
        beast1 = load("Beast_1_animation_1")
        beast2 = load("Beast_1_animation_2")
        beast_frames = aj.pad_to_match([beast1, beast2])
        sprites["beast"] = jnp.concatenate([
            jnp.repeat(beast_frames[0][None], 8, axis=0),
            jnp.repeat(beast_frames[1][None], 8, axis=0),
        ])
        # --------- DIGITS (0-9) ------------------------------------------
        digits_list = [load(f"number_{d}") for d in range(10)]
        # pad_to_match → shape (10, H, W, 4)
        sprites["digits"] = jnp.asarray(aj.pad_to_match(digits_list))
        sprites["life_icon"] = jnp.expand_dims(load("Life"), 0)
        return sprites

    # helper inside SirLancelotRenderer.render()
    def blit(raster, x, y, sprite, *, flip=False):
        # send (y, x) to atraJaxis
        return aj.render_at(raster, y, x, sprite, flip_horizontal=flip)

    # ------------------------------------------------------------------ render
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: GameState):
        raster = jnp.zeros((self.cfg.screen_width, self.cfg.screen_height, 3), dtype=jnp.uint8)

        # Background
        bg_frame = aj.get_sprite_frame(self.sprites["background"], 0)
        raster = aj.render_at(raster, 0, 0, bg_frame)

        # Enemies
        beast_sprite = self.sprites["beast"]
        def draw_enemy(i, rast):
            active = state.enemies.active[i]
            return jax.lax.cond(
                active,
                lambda r: aj.render_at(
                    r,
                    state.enemies.positions[i, 0].astype(jnp.int32),
                    state.enemies.positions[i, 1].astype(jnp.int32),
                    aj.get_sprite_frame(beast_sprite, state.enemies.animation_counter[i]),
                    flip_horizontal=state.enemies.directions[i] < 0,
                ),
                lambda r: r,
                rast,
            )
        raster = jax.lax.fori_loop(0, state.enemies.positions.shape[0], draw_enemy, raster)

        # Player sprite – choose fly sprite if vel_y < 0 (ascending) else neutral
        player_sprite = jax.lax.cond(
            state.player.vel_y < 0,
            lambda: self.sprites["player_fly"],
            lambda: self.sprites["player_neutral"],
        )
        raster = aj.render_at(
            raster,
            state.player.x.astype(jnp.int32),
            state.player.y.astype(jnp.int32),
            aj.get_sprite_frame(player_sprite, 0),
            flip_horizontal=state.player.facing_left,
        )

        # HUD – Score (bottom centre) & lives (bottom left)
        digits_sprite = jnp.asarray(self.sprites["digits"])
        score_digits = aj.int_to_digits(state.score, max_digits=6)  # 6 digits max
        score_x_start = 80 - (len(score_digits) * 4)  # centre
        raster = aj.render_label(raster, score_x_start, self.cfg.screen_height - 18, score_digits, digits_sprite, spacing=8)

        # lives
        life_sprite = aj.get_sprite_frame(self.sprites["life_icon"], 0)
        def draw_life(i, rast):
            return jax.lax.cond(
                i < state.lives,
                lambda r: aj.render_at(r, 10 + i * (life_sprite.shape[0] + 2), self.cfg.screen_height - 18, life_sprite),
                lambda r: r,
                rast,
            )
        raster = jax.lax.fori_loop(0, self.cfg.initial_lives, draw_life, raster)

        # Force leftmost 8‑pixel HUD bar to black (Freeway style for RL cropping)
        raster = raster.at[0:8, :, :].set(0)
        return raster

# -----------------------------------------------------------------------------
# DEMO LOOP (for manual playtesting)
# -----------------------------------------------------------------------------

def get_human_action() -> chex.Array:
    keys = pygame.key.get_pressed()
    left = keys[pygame.K_a] or keys[pygame.K_LEFT]
    right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
    up = keys[pygame.K_w] or keys[pygame.K_UP]
    down = keys[pygame.K_s] or keys[pygame.K_DOWN]
    fire = keys[pygame.K_SPACE]

    x, y = 0, 0
    if left and not right:
        x = -1
    elif right and not left:
        x = 1
    if up and not down:
        y = 1
    elif down and not up:
        y = -1

    # Map to JAX Atari actions (simplified subset)
    if fire:
        if x == -1 and y == 1:
            return jnp.array(Action.UPLEFTFIRE)
        elif x == 1 and y == 1:
            return jnp.array(Action.UPRIGHTFIRE)
        elif x == -1 and y == -1:
            return jnp.array(Action.DOWNLEFTFIRE)
        elif x == 1 and y == -1:
            return jnp.array(Action.DOWNRIGHTFIRE)
        elif y == 1:
            return jnp.array(Action.UPFIRE)
        elif y == -1:
            return jnp.array(Action.DOWNFIRE)
        elif x == -1:
            return jnp.array(Action.LEFTFIRE)
        elif x == 1:
            return jnp.array(Action.RIGHTFIRE)
        else:
            return jnp.array(Action.FIRE)
    else:
        if x == -1 and y == 1:
            return jnp.array(Action.UPLEFT)
        elif x == 1 and y == 1:
            return jnp.array(Action.UPRIGHT)
        elif x == -1 and y == -1:
            return jnp.array(Action.DOWNLEFT)
        elif x == 1 and y == -1:
            return jnp.array(Action.DOWNRIGHT)
        elif x == -1:
            return jnp.array(Action.LEFT)
        elif x == 1:
            return jnp.array(Action.RIGHT)
        elif y == 1:
            return jnp.array(Action.UP)
        elif y == -1:
            return jnp.array(Action.DOWN)
    return jnp.array(Action.NOOP)


def main():
    pygame.init()
    scaling = 4

    env = JaxSirLancelot()
    renderer = SirLancelotRenderer()

    screen = pygame.display.set_mode((env.cfg.screen_width * scaling, env.cfg.screen_height * scaling))
    pygame.display.set_caption("Sir Lancelot ‑ Level 1 (JAX Atari)")

    clock = pygame.time.Clock()
    running = True
    done = False

    jitted_step = jax.jit(env.step)
    jitted_render = jax.jit(renderer.render)
    jitted_reset = jax.jit(env.reset)

    obs, state = jitted_reset()

    while running and not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = get_human_action()
        obs, state, reward, done, info = jitted_step(state, action)
        raster = jitted_render(state)
        aj.update_pygame(screen, raster, scaling, env.cfg.screen_width, env.cfg.screen_height)
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
