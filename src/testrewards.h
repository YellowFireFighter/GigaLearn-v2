#pragma once
// testrewards.h — custom reward classes for staged training
//
// PHASE 1 rewards (already in ExampleMain.cpp):
//   Use these for the first ~2 billion timesteps.
//   Focus on speed, ball contact, and scoring so the bot masters ground play.
//
// PHASE 2 rewards (commented block in EnvCreateFunc):
//   Switch to these once the bot is consistently scoring and around ~500+ MMR.
//   They teach rotation, defense, boost management, and advanced mechanics.
//   Uncomment the Phase 2 block in EnvCreateFunc and comment out Phase 1
//   when you're ready to advance.

#include <RLGymCPP/Rewards/Reward.h>
#include <RLGymCPP/Gamestates/GameState.h>
#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/Math.h>

using namespace RLGC;

// =========================================
// KickoffReward
// =========================================
// PHASE 1 + 2
// More precise than KickoffProximityReward: gives a continuous reward
// proportional to how far ahead of the closest opponent the player is.
class KickoffReward : public Reward {
public:
    float GetReward(const Player& player, const GameState& state, bool isFinal) override {
        // Only active while ball is stationary near centre
        if (state.ball.vel.Length() > 100.f) return 0.f;
        if (state.ball.pos.Length() > 500.f) return 0.f;

        float playerDist = (player.pos - state.ball.pos).Length();
        float closestOpp = 1e9f;

        for (auto& p : state.players) {
            if (p.team != player.team) {
                float d = (p.pos - state.ball.pos).Length();
                if (d < closestOpp) closestOpp = d;
            }
        }

        if (closestOpp >= 1e9f) return 0.f; // no opponents — skip reward

        // Continuous advantage: +1 = very far ahead, -1 = very far behind
        float advantage = (closestOpp - playerDist) / RS_MAX(1.f, closestOpp);
        return RS_CLAMP(advantage, -1.f, 1.f);
    }
};

// =========================================
// ShadowDefenseReward
// =========================================
// PHASE 2
// Rewards the player for positioning between the ball and their own goal
// (the "shadow" position) when the ball is in a threatening location.
// Teaches defensive rotation without just camping in goal.
class ShadowDefenseReward : public Reward {
public:
    float GetReward(const Player& player, const GameState& state, bool isFinal) override {
        Vec ownGoal = (player.team == Team::BLUE)
            ? CommonValues::BLUE_GOAL_BACK
            : CommonValues::ORANGE_GOAL_BACK;

        // Is the ball threatening? (in our half OR ball moving toward our goal)
        bool ballInOurHalf = (player.team == Team::BLUE)
            ? (state.ball.pos.y < 0.f)
            : (state.ball.pos.y > 0.f);

        Vec ballToGoal   = ownGoal - state.ball.pos;
        float ballMovingToGoal = state.ball.vel.Dot(ballToGoal.Normalized());

        if (!ballInOurHalf && ballMovingToGoal <= 0.f) return 0.f;

        // Project the player onto the ball-to-goal line to find the ideal spot
        float lineLen        = ballToGoal.Length();
        Vec   ballToGoalDir  = ballToGoal / RS_MAX(1.f, lineLen);
        Vec   ballToPlayer   = player.pos - state.ball.pos;
        float proj           = RS_CLAMP(ballToPlayer.Dot(ballToGoalDir), 0.f, lineLen);
        Vec   idealPos       = state.ball.pos + ballToGoalDir * proj;

        // How far off the ideal line is the player?
        float lateralErr = (player.pos - idealPos).Length();
        return RS_MAX(0.f, 1.f - lateralErr / 2000.f);
    }
};

// =========================================
// DirectionalStrongTouchReward
// =========================================
// PHASE 2
// Combines StrongTouchReward with direction quality: the ball must be hit
// hard AND toward the opponent's goal for full reward.
class DirectionalStrongTouchReward : public Reward {
public:
    float minRewardedVel, maxRewardedVel;

    DirectionalStrongTouchReward(float minSpeedKPH = 20.f, float maxSpeedKPH = 130.f) {
        minRewardedVel = RLGC::Math::KPHToVel(minSpeedKPH);
        maxRewardedVel = RLGC::Math::KPHToVel(maxSpeedKPH);
    }

    float GetReward(const Player& player, const GameState& state, bool isFinal) override {
        if (!state.prev || !player.ballTouchedStep) return 0.f;

        float hitForce = (state.ball.vel - state.prev->ball.vel).Length();
        if (hitForce < minRewardedVel) return 0.f;
        float touchStrength = RS_MIN(1.f, hitForce / maxRewardedVel);

        // Directionality: how aligned is ball velocity with opponent goal?
        Vec oppGoal = (player.team == Team::BLUE)
            ? CommonValues::ORANGE_GOAL_BACK
            : CommonValues::BLUE_GOAL_BACK;
        Vec ballToGoalDir = (oppGoal - state.ball.pos).Normalized();
        float directionality = RS_MAX(0.f, state.ball.vel.Normalized().Dot(ballToGoalDir));

        return touchStrength * directionality;
    }
};

// =========================================
// FieldRotationReward
// =========================================
// PHASE 2
// Rewards the player for staying in the correct field position relative to
// the ball. Specifically, the player should generally be *behind* the ball
// (between their own goal and the ball) rather than over-committing past it.
// This teaches rotation: don't chase the ball past it.
class FieldRotationReward : public Reward {
public:
    float GetReward(const Player& player, const GameState& state, bool isFinal) override {
        Vec oppGoal = (player.team == Team::BLUE)
            ? CommonValues::ORANGE_GOAL_BACK
            : CommonValues::BLUE_GOAL_BACK;

        float goalY   = oppGoal.y;
        float ballY   = state.ball.pos.y;
        float playerY = player.pos.y;

        // Signed distance: positive means player is behind the ball (good)
        // Blue wants playerY < ballY, Orange wants playerY > ballY
        bool isBlue           = (player.team == Team::BLUE);
        float signedBehind    = isBlue ? (ballY - playerY) : (playerY - ballY);
        float ballToGoalRange = RS_MAX(1.f, fabsf(goalY - ballY));

        // Normalize and cap so the reward stays small
        return RS_CLAMP(signedBehind / ballToGoalRange, -1.f, 1.f) * 0.5f;
    }
};

// =========================================
// StrictDribbleReward
// =========================================
// PHASE 2
// Rewards the player for carrying the ball on top of the car (dribbling).
// The ball needs to be at roof height with low relative velocity to the car.
class StrictDribbleReward : public Reward {
public:
    float GetReward(const Player& player, const GameState& state, bool isFinal) override {
        // Ball should be slightly above the car roof (car roof ~ z+60 above centre)
        float ballZ  = state.ball.pos.z;
        float carZ   = player.pos.z;
        float heightDiff = ballZ - carZ; // positive means ball above car

        if (heightDiff < 60.f || heightDiff > 220.f) return 0.f;

        // Horizontal separation should be small
        float hDist = Vec(
            state.ball.pos.x - player.pos.x,
            state.ball.pos.y - player.pos.y,
            0.f
        ).Length();
        if (hDist > 200.f) return 0.f;

        // Relative velocity between ball and car should be low
        // relVel = Length() — always >= 0, so the division is safe
        float relVel    = (state.ball.vel - player.vel).Length();
        float maxRelVel = RLGC::Math::KPHToVel(20.f);

        return RS_MAX(0.f, 1.f - relVel / maxRelVel);
    }
};

// =========================================
// MawkzyFlickReward
// =========================================
// PHASE 2
// Rewards aerial flicks: the player must be airborne, touch an elevated ball,
// and send it at high speed toward the opponent's goal.
class MawkzyFlickReward : public Reward {
public:
    float GetReward(const Player& player, const GameState& state, bool isFinal) override {
        if (!state.prev || !player.ballTouchedStep) return 0.f;

        // Player must be airborne
        if (player.isOnGround) return 0.f;

        // Ball must have been elevated before contact (above a basic roll height)
        if (state.prev->ball.pos.z < 150.f) return 0.f;

        // Ball velocity toward opponent goal after contact
        Vec oppGoal = (player.team == Team::BLUE)
            ? CommonValues::ORANGE_GOAL_BACK
            : CommonValues::BLUE_GOAL_BACK;
        Vec  ballToGoalDir = (oppGoal - state.ball.pos).Normalized();
        float velToGoal    = state.ball.vel.Dot(ballToGoalDir);

        return RS_MAX(0.f, velToGoal / CommonValues::BALL_MAX_SPEED);
    }
};
