import os
from functools import partial
from typing import NamedTuple, Tuple, List, Dict, Any, Optional

import jax
import jax.numpy as jnp
import chex
# import pygame # Only for demo loop, not core env

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
import jaxatari.rendering.atraJaxis as aj
from jaxatari.renderers import AtraJaxisRenderer

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

class GameConfig(NamedTuple):
    screen_width: int = 160
    screen_height: int = 210 

    top_black_bar_height: int = 24 
    play_area_top_y: int = top_black_bar_height 
    play_area_bottom_y: int = 184 
    lava_contact_y: int = play_area_bottom_y + 2

    player_width: int = 9
    player_height: int = 14
    player_start_x: int = 75 
    player_start_y: int = 130 

    gravity: float = 0.22                  
    flap_impulse: float = -2.8          
    max_fall_speed: float = 2.8     
    max_climb_speed: float = -2.8          
    max_speed_x: float = 1.5 # Max horizontal speed WHEN FLYING
    max_speed_x_grounded: float = 0.5 # Much slower when "walking" if allowed at all
    horizontal_acceleration: float = 0.3   
    horizontal_deceleration: float = 0.4
    ground_friction_deceleration: float = 0.9 # Stronger deceleration on ground   

    enemy_width: int = 9
    enemy_height: int = 12
    enemy_speed: float = 0.85 
    enemy_spawn_y_positions: chex.Array = jnp.array((40, 70, 100, 130), dtype=jnp.float32)
    enemy_spawn_x_buffer: int = 20 
    enemy_respawn_delay_frames: int = 120 

    initial_lives: int = 3
    points_per_enemy: int = 250
    quick_kill_bonus_1: int = 500  
    quick_kill_bonus_2: int = 1000
    quick_kill_bonus_3: int = 1500 
    quick_kill_time_limit: int = 75 

    flap_animation_duration: int = 8 
    flap_cooldown_duration: int = 10 
    death_cooldown_frames: int = 90 

    hud_y_position: int = 8 
    score_x_start: int = 50 
    lives_x_start: int = 10
    digit_sprite_width: int = 8

# -----------------------------------------------------------------------------
# STATE REPRESENTATIONS
# -----------------------------------------------------------------------------

class PlayerState(NamedTuple):
    x: chex.Array
    y: chex.Array
    vel_x: chex.Array
    vel_y: chex.Array
    facing_left: chex.Array 
    flap_cooldown: chex.Array
    is_flapping_sprite: chex.Array
    flap_animation_timer: chex.Array
    is_dead_falling: chex.Array 
    death_cooldown: chex.Array
    is_grounded: chex.Array # New state to track if player is on the ground

class EnemyState(NamedTuple):
    positions: chex.Array 
    directions: chex.Array 
    active: chex.Array 
    animation_counter: chex.Array 
    respawn_cooldown: chex.Array

class GameState(NamedTuple):
    player: PlayerState
    enemies: EnemyState
    score: chex.Array
    lives: chex.Array
    time: chex.Array
    game_over: chex.Array
    last_kill_time: chex.Array
    kills_in_quick_succession: chex.Array
    key: jax.random.PRNGKey


# -----------------------------------------------------------------------------
# OBSERVATION / INFO
# -----------------------------------------------------------------------------

class EntityPosition(NamedTuple): 
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    facing_left: Optional[jnp.ndarray] = None

class SirLancelotObservation(NamedTuple):
    player: EntityPosition
    enemies: jnp.ndarray 
    score: jnp.ndarray
    lives: jnp.ndarray

class SirLancelotInfo(NamedTuple):
    time: jnp.ndarray
    quick_kills: jnp.ndarray

# -----------------------------------------------------------------------------
# ENVIRONMENT IMPLEMENTATION
# -----------------------------------------------------------------------------

class JaxSirLancelot(JaxEnvironment[GameState, SirLancelotObservation, SirLancelotInfo]):

    def __init__(self, max_enemies: int = 4): 
        super().__init__()
        self.cfg = GameConfig()
        self.max_enemies = max_enemies

    def _spawn_single_enemy(self, key: jax.random.PRNGKey, enemy_idx_dummy: int, cfg_tuple: GameConfig):
        key_y, key_offset = jax.random.split(key, 2)

        enemy_spawn_y_positions_arr = jnp.asarray(cfg_tuple.enemy_spawn_y_positions, dtype=jnp.float32)
        # guarantee each enemy slot i gets a distinct preset height
        spawn_y = enemy_spawn_y_positions_arr[ enemy_idx_dummy % enemy_spawn_y_positions_arr.shape[0] ]

        x_offset = cfg_tuple.enemy_spawn_x_buffer + jax.random.uniform(key_offset, minval=0, maxval=10)
        spawn_x = cfg_tuple.screen_width + x_offset  # always start off‑screen right
        direction = -1                               # always fly left

        return jnp.array([spawn_x, spawn_y], dtype=jnp.float32), jnp.array(direction, dtype=jnp.int32)


    def reset(self, key: jax.random.PRNGKey = None):
        cfg = self.cfg 
        if key is None: key = jax.random.PRNGKey(0)
        
        key_player, key_enemies_master, key_state_global = jax.random.split(key, 3)

        player = PlayerState(
            x=jnp.array(cfg.player_start_x, dtype=jnp.float32),
            y=jnp.array(cfg.player_start_y, dtype=jnp.float32),
            vel_x=jnp.array(0.0, dtype=jnp.float32),
            vel_y=jnp.array(0.0, dtype=jnp.float32),
            facing_left=jnp.array(False), 
            flap_cooldown=jnp.array(0, dtype=jnp.int32),
            is_flapping_sprite=jnp.array(False),
            flap_animation_timer=jnp.array(0, dtype=jnp.int32),
            is_dead_falling=jnp.array(False),
            death_cooldown=jnp.array(0, dtype=jnp.int32),
            is_grounded=jnp.array(False) # Initialize is_grounded
        )
        
        keys_enemies_spawn = jax.random.split(key_enemies_master, self.max_enemies)
        dummy_indices = jnp.arange(self.max_enemies)

        all_positions, all_directions = jax.vmap(self._spawn_single_enemy, in_axes=(0, 0, None))(
            keys_enemies_spawn, dummy_indices, self.cfg
        )
        
        enemies = EnemyState(
            positions=all_positions, 
            directions=all_directions,
            active=jnp.ones((self.max_enemies,), dtype=bool), 
            animation_counter=jnp.zeros((self.max_enemies,), dtype=jnp.int32),
            respawn_cooldown=jnp.zeros((self.max_enemies,), dtype=jnp.int32)
        )

        state = GameState(
            player=player, enemies=enemies, score=jnp.array(0, dtype=jnp.int32),
            lives=jnp.array(cfg.initial_lives, dtype=jnp.int32), time=jnp.array(0, dtype=jnp.int32),
            game_over=jnp.array(False),
            last_kill_time=jnp.array(-cfg.quick_kill_time_limit -1, dtype=jnp.int32), 
            kills_in_quick_succession=jnp.array(0, dtype=jnp.int32),
            key = key_state_global
        )
        return self._get_observation(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: GameState, action: int):
        cfg = self.cfg
        key, key_player_respawn, key_enemy_respawn_master = jax.random.split(state.key, 3)
        
        current_player_state = state.player

        def update_alive_player_physics(p_state, act): 
            move_left_input = jnp.any(jnp.array([act == Action.LEFT, act == Action.UPLEFT, act == Action.DOWNLEFT, act == Action.LEFTFIRE, act == Action.UPLEFTFIRE, act == Action.DOWNLEFTFIRE]))
            move_right_input = jnp.any(jnp.array([act == Action.RIGHT, act == Action.UPRIGHT, act == Action.DOWNRIGHT, act == Action.RIGHTFIRE, act == Action.UPRIGHTFIRE, act == Action.DOWNRIGHTFIRE]))

            target_vel_x = 0.0
            target_vel_x = jnp.where(move_left_input, -cfg.max_speed_x, target_vel_x)
            target_vel_x = jnp.where(move_right_input, cfg.max_speed_x, target_vel_x)
            
            new_facing_left = jax.lax.cond(
                move_left_input, 
                lambda _: True, 
                lambda _: jax.lax.cond(
                    move_right_input, 
                    lambda __: False, 
                    lambda __: p_state.facing_left, 
                    None 
                ), 
                None 
            )
            
            # Grounded check (player's feet are at the bottom of play area)
            # A small tolerance (e.g., 1 pixel) might be needed for float precision
            is_on_ground_check = (p_state.y + cfg.player_height) >= cfg.play_area_bottom_y 
            
            # --- Horizontal Movement ---
            current_vel_x = p_state.vel_x
            
            # Determine max speed and deceleration based on grounded state
            # If grounded, horizontal input might be ignored or heavily dampened unless flapping
            allow_ground_control = False # Set to True if you want some ground movement, False for Joust-like stop
            
            effective_max_speed_x = jnp.where(is_on_ground_check & ~allow_ground_control, 0.0, cfg.max_speed_x) # No speed on ground if not allowed
            effective_deceleration = jnp.where(is_on_ground_check, cfg.ground_friction_deceleration, cfg.horizontal_deceleration)
            
            effective_target_vel_x = 0.0
            effective_target_vel_x = jnp.where(move_left_input, -effective_max_speed_x, effective_target_vel_x)
            effective_target_vel_x = jnp.where(move_right_input, effective_max_speed_x, effective_target_vel_x)

            is_reversing = (move_left_input & (current_vel_x > 0.01)) | (move_right_input & (current_vel_x < -0.01))
            
            new_vel_x = current_vel_x
            new_vel_x = jnp.where(is_reversing, 0.0, new_vel_x) 
            
            # Apply acceleration or deceleration
            # If grounded and no ground control, target is 0, so it will decelerate via friction
            new_vel_x = jnp.where(effective_target_vel_x != 0.0,
                                  jnp.clip(new_vel_x + jnp.sign(effective_target_vel_x) * cfg.horizontal_acceleration, -effective_max_speed_x, effective_max_speed_x),
                                  new_vel_x * (1.0 - effective_deceleration)
                                 )
            # If grounded and not flapping, force vel_x towards 0 more aggressively
            new_vel_x = jnp.where(is_on_ground_check & (p_state.flap_cooldown > 0), new_vel_x * (1.0 - cfg.ground_friction_deceleration * 2.0), new_vel_x)


            # --- Vertical Movement (Flapping & Gravity) ---
            fire_pressed = jnp.any(jnp.array([act == Action.FIRE, act == Action.UPFIRE, act == Action.DOWNFIRE, act == Action.LEFTFIRE, act == Action.RIGHTFIRE, act == Action.UPLEFTFIRE, act == Action.UPRIGHTFIRE, act == Action.DOWNLEFTFIRE, act == Action.DOWNRIGHTFIRE]))
            can_flap = p_state.flap_cooldown == 0
            is_flapping_now = fire_pressed & can_flap

            new_vel_y = p_state.vel_y + cfg.gravity
            new_vel_y = jnp.where(is_flapping_now, cfg.flap_impulse, new_vel_y)
            new_vel_y = jnp.clip(new_vel_y, cfg.max_climb_speed, cfg.max_fall_speed)

            # If grounded and not trying to flap up, stop vertical velocity
            new_vel_y = jnp.where(is_on_ground_check & (new_vel_y > 0) & ~is_flapping_now, 0.0, new_vel_y)


            new_flap_cooldown = jnp.where(is_flapping_now, cfg.flap_cooldown_duration, jnp.maximum(p_state.flap_cooldown - 1, 0))
            new_flap_animation_timer = jnp.where(is_flapping_now, cfg.flap_animation_duration, jnp.maximum(0, p_state.flap_animation_timer - 1))
            
            raw_player_x = p_state.x + new_vel_x
            new_player_x = jnp.where(raw_player_x < -cfg.player_width + 1, cfg.screen_width -1, 
                                 jnp.where(raw_player_x > cfg.screen_width -1, -cfg.player_width + 1, raw_player_x))
            
            # Apply vertical movement and clip to play area
            new_player_y_candidate = p_state.y + new_vel_y
            new_player_y = jnp.clip(new_player_y_candidate, cfg.play_area_top_y, cfg.play_area_bottom_y - cfg.player_height)
            
            # Update is_grounded state
            new_is_grounded = (new_player_y + cfg.player_height) >= cfg.play_area_bottom_y
            
            return p_state._replace(
                x=new_player_x, y=new_player_y, vel_x=new_vel_x, vel_y=new_vel_y,
                facing_left=new_facing_left, flap_cooldown=new_flap_cooldown,
                is_flapping_sprite=new_flap_animation_timer > 0, flap_animation_timer=new_flap_animation_timer,
                is_grounded=new_is_grounded
            )

        def update_dead_player_physics(p_state, act_dummy): 
            new_player_y = p_state.y + cfg.max_fall_speed 
            new_death_cooldown = jnp.maximum(0, p_state.death_cooldown - 1)
            return p_state._replace(y=new_player_y, death_cooldown=new_death_cooldown, is_grounded=False) # Not grounded when dead falling

        player_after_movement = jax.lax.cond(
            current_player_state.is_dead_falling,
            update_dead_player_physics,
            update_alive_player_physics,
            current_player_state, action
        )
        
        player_feet_y = player_after_movement.y + cfg.player_height
        fell_into_lava = player_feet_y >= cfg.lava_contact_y
        
        player_became_dead_this_step = fell_into_lava & ~current_player_state.is_dead_falling
        player_state_after_env_effects = player_after_movement._replace(
            is_dead_falling = player_after_movement.is_dead_falling | player_became_dead_this_step,
            death_cooldown = jnp.where(player_became_dead_this_step, cfg.death_cooldown_frames, player_after_movement.death_cooldown),
            vel_x = jnp.where(player_became_dead_this_step, 0.0, player_after_movement.vel_x),
            vel_y = jnp.where(player_became_dead_this_step, 0.0, player_after_movement.vel_y)
        )

        current_enemies_state = state.enemies
        def move_single_enemy(idx_move, current_es_state_for_move):
            pos = current_es_state_for_move.positions[idx_move]
            dir_ = current_es_state_for_move.directions[idx_move]
            active = current_es_state_for_move.active[idx_move]
            anim = current_es_state_for_move.animation_counter[idx_move]
            cooldown = current_es_state_for_move.respawn_cooldown[idx_move]

            new_cooldown = jnp.maximum(0, cooldown - 1)

            # small deterministic speed variation so they don't stay in formation
            speed_variation = cfg.enemy_speed * (1.0 + 0.05 * idx_move)

            def _move_active_enemy_logic_closure_fixed(_op):
                moved_x = pos[0] + dir_ * speed_variation

                wrapped_x = jnp.where(
                    moved_x < -cfg.enemy_width,
                    cfg.screen_width + cfg.enemy_width,
                    jnp.where(
                        moved_x > cfg.screen_width + cfg.enemy_width,
                        -cfg.enemy_width,
                        moved_x,
                    ),
                )
                new_pos = jnp.array([wrapped_x, pos[1]])
                return new_pos, dir_, (anim + 1) % 16, active, new_cooldown

            def _passthrough_inactive_enemy_fixed(_operand):
                return pos, dir_, anim, active, new_cooldown

            new_pos, new_dir, new_anim, new_active_flag, new_cooldown_out = jax.lax.cond(
                active,
                _move_active_enemy_logic_closure_fixed,
                _passthrough_inactive_enemy_fixed,
                None,
            )
            return new_pos, new_dir, new_active_flag, new_anim, new_cooldown_out

        def enemy_move_loop_body(i_loop, es_carry_loop):
            p_loop, d_loop, act_loop, anim_c_loop, rc_loop = move_single_enemy(i_loop, es_carry_loop)
            return es_carry_loop._replace(
                positions=es_carry_loop.positions.at[i_loop].set(p_loop),
                directions=es_carry_loop.directions.at[i_loop].set(d_loop),
                active=es_carry_loop.active.at[i_loop].set(act_loop),
                animation_counter=es_carry_loop.animation_counter.at[i_loop].set(anim_c_loop),
                respawn_cooldown=es_carry_loop.respawn_cooldown.at[i_loop].set(rc_loop)
            )
        enemies_after_movement = jax.lax.fori_loop(0, self.max_enemies, enemy_move_loop_body, current_enemies_state)

        def perform_combat_resolution(p_combat_state, e_combat_state, current_score, current_lives, current_lkt, current_kqs):
            p_rect_l, p_rect_r = p_combat_state.x, p_combat_state.x + cfg.player_width
            p_rect_t, p_rect_b = p_combat_state.y, p_combat_state.y + cfg.player_height

            e_rect_l, e_rect_r = e_combat_state.positions[:,0], e_combat_state.positions[:,0] + cfg.enemy_width
            e_rect_t, e_rect_b = e_combat_state.positions[:,1], e_combat_state.positions[:,1] + cfg.enemy_height

            coll_x = (p_rect_l < e_rect_r) & (p_rect_r > e_rect_l)
            coll_y = (p_rect_t < e_rect_b) & (p_rect_b > e_rect_t)
            base_coll_mask = e_combat_state.active & coll_x & coll_y

            # sprite image faces LEFT by default; we flip when looking right
            facing_left_actual = p_combat_state.facing_left

            player_facing_enemy = jnp.where(
                facing_left_actual,
                e_combat_state.positions[:, 0] < p_combat_state.x,
                e_combat_state.positions[:, 0] > p_combat_state.x,
            )

            enemy_facing_player = jnp.where(
                e_combat_state.directions == -1,
                p_combat_state.x < e_combat_state.positions[:, 0],
                p_combat_state.x > e_combat_state.positions[:, 0],
            )

            both_facing  = player_facing_enemy & enemy_facing_player
            player_higher = p_combat_state.y < e_combat_state.positions[:, 1]
            enemy_higher  = p_combat_state.y > e_combat_state.positions[:, 1]

            player_wins = (player_facing_enemy & ~enemy_facing_player) | (both_facing & player_higher)
            enemy_wins  = (~player_facing_enemy & enemy_facing_player) | (both_facing & enemy_higher)
            # if heights equal and both facing, it's a tie → neither wins
            player_defeats_enemy_conditions = base_coll_mask & player_wins
            enemy_defeats_player_conditions = base_coll_mask & enemy_wins

            num_player_wins_this_step = jnp.sum(player_defeats_enemy_conditions)
            num_enemy_wins_this_step = jnp.sum(enemy_defeats_player_conditions)

            new_score_val = current_score
            new_lkt_val = current_lkt
            new_kqs_val = current_kqs

            def apply_bonus_for_kills_closure(carry_bonus):
                s_bonus, lkt_bonus, kqs_bonus = carry_bonus
                time_now = state.time
                time_since_last = time_now - lkt_bonus
                is_quick = time_since_last <= cfg.quick_kill_time_limit

                kqs_for_bonus_calc = jnp.where(is_quick, kqs_bonus, 0)
                bonus_idx = jnp.minimum(kqs_for_bonus_calc, 2)
                bonus_val = jax.lax.switch(bonus_idx, [
                    lambda: jnp.array(cfg.quick_kill_bonus_1, dtype=jnp.int32),
                    lambda: jnp.array(cfg.quick_kill_bonus_2, dtype=jnp.int32),
                    lambda: jnp.array(cfg.quick_kill_bonus_3, dtype=jnp.int32)
                ])
                score_update = (cfg.points_per_enemy * num_player_wins_this_step) + bonus_val
                return s_bonus + score_update, time_now, kqs_for_bonus_calc + num_player_wins_this_step

            new_score_val, new_lkt_val, new_kqs_val = jax.lax.cond(
                num_player_wins_this_step > 0,
                apply_bonus_for_kills_closure,
                lambda carry_passthrough: carry_passthrough,
                (new_score_val, new_lkt_val, new_kqs_val)
            )

            updated_enemy_active = e_combat_state.active & ~player_defeats_enemy_conditions
            player_hit_in_combat = num_enemy_wins_this_step > 0

            final_p_state = p_combat_state._replace(
                is_dead_falling=p_combat_state.is_dead_falling | player_hit_in_combat,
                death_cooldown=jnp.where(player_hit_in_combat, cfg.death_cooldown_frames, p_combat_state.death_cooldown),
                vel_x=jnp.where(player_hit_in_combat, 0.0, p_combat_state.vel_x),
                vel_y=jnp.where(player_hit_in_combat, 0.0, p_combat_state.vel_y)
            )
            final_e_state = e_combat_state._replace(active=updated_enemy_active)
            final_lives = current_lives - jnp.where(player_hit_in_combat, 1, 0)

            return final_p_state, final_e_state, new_score_val, final_lives, new_lkt_val, new_kqs_val

        player_after_combat, enemies_after_combat, score_after_combat, lives_after_combat, lkt_after_combat, kqs_after_combat = jax.lax.cond(
            ~player_state_after_env_effects.is_dead_falling,
            lambda: perform_combat_resolution(player_state_after_env_effects, enemies_after_movement, state.score, state.lives, state.last_kill_time, state.kills_in_quick_succession),
            lambda: (player_state_after_env_effects, enemies_after_movement, state.score, state.lives, state.last_kill_time, state.kills_in_quick_succession)
        )
        
        keys_enemy_respawn = jax.random.split(key_enemy_respawn_master, self.max_enemies)
        current_e_state_for_respawn = enemies_after_combat
        defeated_this_step_mask = enemies_after_movement.active & ~enemies_after_combat.active
        
        def respawn_loop_body(i_respawn, es_respawn_carry):
            current_cooldown = es_respawn_carry.respawn_cooldown[i_respawn]
            is_defeated_now = defeated_this_step_mask[i_respawn]
            
            cooldown_after_defeat = jnp.where(is_defeated_now, cfg.enemy_respawn_delay_frames, current_cooldown)
            
            can_respawn_slot = (~es_respawn_carry.active[i_respawn]) & (cooldown_after_defeat == 0)
            # Level 1: no new snakes once all four are defeated
            should_respawn_this_enemy = False

            pos_after_spawn, dir_after_spawn = self._spawn_single_enemy(keys_enemy_respawn[i_respawn], i_respawn, self.cfg)
            
            new_pos_i = jnp.where(should_respawn_this_enemy, pos_after_spawn, es_respawn_carry.positions[i_respawn])
            new_dir_i = jnp.where(should_respawn_this_enemy, dir_after_spawn, es_respawn_carry.directions[i_respawn])
            new_active_i = jnp.where(should_respawn_this_enemy, True, es_respawn_carry.active[i_respawn])
            new_anim_counter_i = jnp.where(should_respawn_this_enemy, 0, es_respawn_carry.animation_counter[i_respawn])

            return es_respawn_carry._replace(
                positions=es_respawn_carry.positions.at[i_respawn].set(new_pos_i),
                directions=es_respawn_carry.directions.at[i_respawn].set(new_dir_i),
                active=es_respawn_carry.active.at[i_respawn].set(new_active_i),
                respawn_cooldown=es_respawn_carry.respawn_cooldown.at[i_respawn].set(cooldown_after_defeat),
                animation_counter=es_respawn_carry.animation_counter.at[i_respawn].set(new_anim_counter_i)
            )
        enemies_final_state = jax.lax.fori_loop(0, self.max_enemies, respawn_loop_body, current_e_state_for_respawn)
        
        player_death_sequence_complete = player_after_combat.is_dead_falling & (player_after_combat.death_cooldown == 0)
        final_player_state_for_step = player_after_combat
        final_game_over_state = state.game_over 

        def handle_player_death_completion_closure(p_state_death, lvs_death):
            can_respawn_flag = lvs_death >= 0 
            p_respawned = p_state_death._replace( 
                x=jnp.array(cfg.player_start_x, dtype=jnp.float32),
                y=jnp.array(cfg.player_start_y, dtype=jnp.float32),
                vel_x=0.0, vel_y=0.0, is_dead_falling=False, flap_cooldown=0,
                facing_left=False, is_flapping_sprite=False, flap_animation_timer=0, death_cooldown=0,
                is_grounded=False # Reset grounded on respawn
            )
            final_p_after_death_seq = jax.lax.cond(can_respawn_flag, lambda _: p_respawned, lambda _: p_state_death, None) 
            new_go_state_after_death_seq = ~can_respawn_flag 
            return final_p_after_death_seq, new_go_state_after_death_seq

        final_player_state_for_step, final_game_over_state = jax.lax.cond(
            player_death_sequence_complete,
            lambda operand_tuple: handle_player_death_completion_closure(operand_tuple[0], operand_tuple[1]), 
            lambda operand_tuple: (operand_tuple[0], operand_tuple[2]), 
            (player_after_combat, lives_after_combat, state.game_over) 
        )

        # if every enemy is inactive, the player has cleared level 1
        all_enemies_defeated = jnp.all(~enemies_final_state.active)
        final_game_over_state = final_game_over_state | all_enemies_defeated

        new_state = GameState(
            player=final_player_state_for_step, enemies=enemies_final_state, 
            score=score_after_combat, lives=lives_after_combat,
            time=state.time + 1, game_over=final_game_over_state,
            last_kill_time=lkt_after_combat,
            kills_in_quick_succession=kqs_after_combat,
            key=key 
        )

        reward = score_after_combat - state.score 
        done = self._get_done(new_state)
        obs = self._get_observation(new_state)
        info = self._get_info(new_state)

        return obs, new_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: GameState):
        player_ent = EntityPosition(
            x=state.player.x, y=state.player.y,
            width=jnp.array(self.cfg.player_width, dtype=jnp.float32), 
            height=jnp.array(self.cfg.player_height, dtype=jnp.float32),
            facing_left=state.player.facing_left
        )
        
        enemy_widths = jnp.full((self.max_enemies,), self.cfg.enemy_width, dtype=jnp.float32)
        enemy_heights = jnp.full((self.max_enemies,), self.cfg.enemy_height, dtype=jnp.float32)
        
        enemy_obs_data = jnp.stack([
            state.enemies.positions[:,0], state.enemies.positions[:,1],
            enemy_widths, enemy_heights,
            state.enemies.active.astype(jnp.float32),
            state.enemies.directions.astype(jnp.float32) 
        ], axis=-1) 

        return SirLancelotObservation(
            player=player_ent, enemies=enemy_obs_data,
            score=state.score, lives=state.lives
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: GameState):
        return SirLancelotInfo(time=state.time, quick_kills=state.kills_in_quick_succession)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: GameState):
        return state.game_over

    @partial(jax.jit, static_argnums=(0,))
    def get_action_space(self):
        return Action.get_all_values()


class SirLancelotRenderer(AtraJaxisRenderer):
    def __init__(self):
        super().__init__()
        self.cfg = GameConfig()
        self.sprites = self._load_sprites()
        self.max_enemies = 4 

    def _load_sprites(self) -> Dict[str, Any]: 
        MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(MODULE_DIR, "sprites/sir_lancelot/"),
            os.path.join(MODULE_DIR, "../sprites/sir_lancelot/"),
            os.path.join(os.getcwd(), "src/jaxatari/games/sprites/sir_lancelot/"),
            os.path.join(os.getcwd(), "sprites/sir_lancelot/") 
        ]
        sprite_path = ""
        for p_val in possible_paths: 
            if os.path.exists(os.path.join(p_val, "Background_lvl1.npy")):
                sprite_path = p_val
                break
        if not sprite_path: raise FileNotFoundError(f"Sprite directory for SirLancelot not found in checked paths: {possible_paths}")
        
        def load_s(name: str): return aj.loadFrame(os.path.join(sprite_path, name + ".npy")).astype(jnp.uint8)

        sprites: Dict[str, Any] = {}
        
        raw_bg = load_s("Background_lvl1") 
        bg_w, bg_h_raw, bg_c = raw_bg.shape
        if bg_h_raw > self.cfg.screen_height:
            cropped_bg = raw_bg[:, :self.cfg.screen_height, :]
        elif bg_h_raw < self.cfg.screen_height:
            padding_height = self.cfg.screen_height - bg_h_raw
            cropped_bg = jnp.pad(raw_bg, ((0,0), (0, padding_height), (0,0)), mode='constant', constant_values=0)
        else:
            cropped_bg = raw_bg
        sprites["background"] = jnp.expand_dims(cropped_bg, 0)

        player_frames_list = aj.pad_to_match([load_s("SirLancelot_lvl1_neutral"), load_s("SirLancelot_lvl1_fly")])
        sprites["player_neutral"] = jnp.expand_dims(player_frames_list[0], 0)
        sprites["player_fly"] = jnp.expand_dims(player_frames_list[1], 0)

        beast_frames_list = aj.pad_to_match([load_s("Beast_1_animation_1"), load_s("Beast_1_animation_2")])
        sprites["beast"] = jnp.concatenate([ 
            jnp.repeat(jnp.expand_dims(beast_frames_list[0],0), 8, axis=0),
            jnp.repeat(jnp.expand_dims(beast_frames_list[1],0), 8, axis=0),
        ])

        digits_list_loaded = [load_s(f"number_{d}") for d in range(10)]
        padded_digits_list = aj.pad_to_match(digits_list_loaded)
        sprites["digits"] = jnp.array(padded_digits_list) 
        
        life_icon_loaded_list = aj.pad_to_match([load_s("Life")])
        sprites["life_icon"] = jnp.array(life_icon_loaded_list) 
        return sprites

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: GameState):
        raster = jnp.zeros((self.cfg.screen_width, self.cfg.screen_height, 3), dtype=jnp.uint8)
        
        bg_frame = aj.get_sprite_frame(self.sprites["background"], 0) 
        raster = aj.render_at(raster, 0, 0, bg_frame)

        beast_sprite_sheet = self.sprites["beast"] 
        def draw_enemy_loop_body(idx, current_raster_enemies): 
            enemy_active = state.enemies.active[idx]
            enemy_x = state.enemies.positions[idx, 0].astype(jnp.int32)
            enemy_y = state.enemies.positions[idx, 1].astype(jnp.int32)
            flip = state.enemies.directions[idx] == -1 
            anim_idx = state.enemies.animation_counter[idx]
            enemy_frame = aj.get_sprite_frame(beast_sprite_sheet, anim_idx) 
            return jax.lax.cond(enemy_active, lambda r_enemy: aj.render_at(r_enemy, enemy_x, enemy_y, enemy_frame, flip_horizontal=flip), lambda r_enemy: r_enemy, current_raster_enemies)
        
        raster = jax.lax.fori_loop(0, self.max_enemies, draw_enemy_loop_body, raster) 
        
        def render_player_fn_closure(r_inner_player): 
            player_sprite_sheet_frames = jax.lax.cond(
                state.player.is_flapping_sprite, 
                lambda _: self.sprites["player_fly"], 
                lambda _: self.sprites["player_neutral"],
                None 
            )
            player_frame_to_draw = aj.get_sprite_frame(player_sprite_sheet_frames, 0) 
            return aj.render_at(
                r_inner_player,
                state.player.x.astype(jnp.int32),
                state.player.y.astype(jnp.int32),
                player_frame_to_draw,
                flip_horizontal=jnp.logical_not(state.player.facing_left),  # flip when looking right
            )
        
        should_render_player = ~((state.player.is_dead_falling & (state.player.death_cooldown == 0)) & (state.lives < 0))
        raster = jax.lax.cond(should_render_player, render_player_fn_closure, lambda r_passthrough_player: r_passthrough_player, raster)

        raster = raster.at[:, 0:self.cfg.top_black_bar_height, :].set(0) 

        # Always show six digits (leading zeros) – all JAX ops, no Python int
        score_val   = state.score.astype(jnp.int32)
        place_vals  = jnp.array([100000, 10000, 1000, 100, 10, 1], dtype=jnp.int32)
        score_digits = jnp.mod(score_val // place_vals, 10)  # shape (6,)

        digit_w      = self.sprites["digits"].shape[1]
        score_total  = 6 * digit_w + 5                       # five 1-px gaps
        score_x0     = (self.cfg.screen_width - score_total) // 2

        def _draw_digit(i, rast):
            x = score_x0 + i * (digit_w + 1)
            idx = score_digits[i]
            return aj.render_at(rast, x, self.cfg.hud_y_position, self.sprites["digits"][idx])

        raster = jax.lax.fori_loop(0, 6, _draw_digit, raster)

        life_icon_frame = aj.get_sprite_frame(self.sprites["life_icon"], 0) 
        life_icon_w = life_icon_frame.shape[0] 
        def draw_life_loop_body(idx, current_raster_lives_loop): 
            x_pos = self.cfg.lives_x_start + idx * (life_icon_w + 2) 
            return jax.lax.cond(idx < state.lives, lambda r_life: aj.render_at(r_life, x_pos, self.cfg.hud_y_position, life_icon_frame), lambda r_life: r_life, current_raster_lives_loop)
        raster = jax.lax.fori_loop(0, self.cfg.initial_lives, draw_life_loop_body, raster)
        
        return raster

if __name__ == "__main__":
    import pygame 

    pygame.init()
    cfg = GameConfig()
    scaling = int(3) 

    env = JaxSirLancelot()
    renderer = SirLancelotRenderer()

    screen = pygame.display.set_mode((cfg.screen_width * scaling, cfg.screen_height * scaling))
    pygame.display.set_caption("Sir Lancelot - Level 1 (JAX Atari)")
    clock = pygame.time.Clock()
    running = True

    key = jax.random.PRNGKey(0) 
    obs, state = env.reset(key=key) 

    jitted_step = jax.jit(env.step)
    jitted_render = jax.jit(renderer.render)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                key, reset_key_sub = jax.random.split(state.key) 
                obs, state = env.reset(key=reset_key_sub)
                print("Game Reset!")
        
        pressed_pygame_keys = pygame.key.get_pressed()
        current_action = Action.NOOP 
        if pressed_pygame_keys[pygame.K_LEFT]: current_action = Action.LEFT
        elif pressed_pygame_keys[pygame.K_RIGHT]: current_action = Action.RIGHT
        if pressed_pygame_keys[pygame.K_SPACE]: 
            if current_action == Action.LEFT: current_action = Action.LEFTFIRE
            elif current_action == Action.RIGHT: current_action = Action.RIGHTFIRE
            else: current_action = Action.FIRE
        
        obs, state, reward, done, info = jitted_step(state, jnp.array(current_action, dtype=jnp.int32))

        if reward != 0: print(f"Score: {state.score}, Reward: {reward}, Lives: {state.lives}") 
        if done:
            print(f"Game Over! Final Score: {state.score}, Time: {state.time}, Quick Kills: {info.quick_kills}")
            key, reset_key_sub_done = jax.random.split(state.key)
            obs, state = env.reset(key=reset_key_sub_done)

        raster_frame = jitted_render(state) 
        aj.update_pygame(screen, raster_frame, scaling, cfg.screen_width, cfg.screen_height)
        
        pygame.display.flip()
        clock.tick(60) 

    pygame.quit()