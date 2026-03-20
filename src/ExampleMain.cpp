#include <GigaLearnCPP/Learner.h>
#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/ObsBuilders/DefaultObs.h>
#include <RLGymCPP/StateSetters/KickoffState.h>
#include <RLGymCPP/StateSetters/RandomState.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>
#include <RLGymCPP/Rewards/Reward.h>
#include "testrewards.h"

using namespace GGL;
using namespace RLGC;

// =========================================
// SpeedTowardBallReward
// =========================================
class SpeedTowardBallReward : public Reward {
public:
    float GetReward(const Player& player, const GameState& state, bool isFinal) override {
        Vec posDiff = state.ball.pos - player.pos;
        float dist = posDiff.Length();
        if (dist < 1.f) return 0.f;
        Vec dirToBall = posDiff / dist;
        float speedToward = player.vel.Dot(dirToBall);
        return RS_MAX(0.f, speedToward / CommonValues::CAR_MAX_SPEED);
    }
};

// =========================================
// VelocityBallToGoalOnTouchReward
// =========================================
class VelocityBallToGoalOnTouchReward : public Reward {
public:
    float GetReward(const Player& player, const GameState& state, bool isFinal) override {
        if (!player.ballTouchedStep) return 0.f;
        float goalY = (player.team == Team::ORANGE) ? -CommonValues::BACK_NET_Y : CommonValues::BACK_NET_Y;
        Vec posDiff = Vec(0, goalY, 0) - state.ball.pos;
        float dist = posDiff.Length();
        if (dist < 1.f) return 0.f;
        Vec dirToGoal = posDiff / dist;
        float velToward = state.ball.vel.Dot(dirToGoal);
        return RS_MAX(0.f, velToward / CommonValues::BALL_MAX_SPEED);
    }
};

// =========================================
// KickoffProximityReward
// =========================================
class KickoffProximityReward : public Reward {
public:
    float GetReward(const Player& player, const GameState& state, bool isFinal) override {
        if (state.ball.vel.Length() > 100.f) return 0.f;

        float playerDist = (player.pos - state.ball.pos).Length();
        float closestOpp = 1e9f;

        for (auto& p : state.players) {
            if (p.team != player.team) {
                float d = (p.pos - state.ball.pos).Length();
                if (d < closestOpp) closestOpp = d;
            }
        }

        if (closestOpp >= 1e9f) return 0.f;
        return (playerDist < closestOpp) ? 1.f : -1.f;
    }
};

// =========================================
// RandomStateSetter (80% kickoff, 20% random)
// =========================================
class RandomStateSetter : public StateSetter {
public:
    KickoffState kickoff;
    RandomState random;

    RandomStateSetter() : kickoff(), random(true, true, false) {}

    void ResetArena(Arena* arena) override {
        if ((rand() % 10) < 8)
            kickoff.ResetArena(arena);
        else
            random.ResetArena(arena);
    }
};

// =========================================
// Env Create Function
// =========================================
EnvCreateResult EnvCreateFunc(int index) {

    // ── PHASE 1 rewards (use until ~2 billion timesteps) ──────────────────
    // Focuses on speed, ball contact, and scoring.  Master ground play first.
    // NOTE: If the bot drives slowly, the SpeedReward is the fix — it gives a
    // constant incentive to go fast at all times, not just when chasing the ball.
    std::vector<WeightedReward> rewards = {
        // Always go fast — the main fix for slow driving.
        // SpeedReward is defined in CommonRewards.h: returns vel.Length() / CAR_MAX_SPEED.
        { new SpeedReward(),                      1.0f   },
        // Small orientation signal — don't let this dominate
        { new FaceBallReward(),                   0.05f  },
        // Getting to the ball (reduced so the bot doesn't just ball-chase)
        { new SpeedTowardBallReward(),             2.0f   },
        // Strongly reward shooting toward the opponent goal on contact
        { new VelocityBallToGoalOnTouchReward(),  12.0f  },
        // Reward hitting the ball hard — discourages gentle nudges
        { new StrongTouchReward(),                 8.0f   },
        // Scoring is the primary objective
        { new GoalReward(),                       1000.0f },
        // Win kickoffs — getting to the ball first matters
        { new KickoffProximityReward(),            5.0f   },
        // Tiny ball-touch reward so the bot still learns contact basics
        { new TouchBallReward(),                   1.0f   },
        // Restored: small air reward encourages the bot not to be flat-footed
        { new AirReward(),                         0.15f  },
    };

    // ── PHASE 2 rewards (swap in after ~2B timesteps / ~500+ MMR) ─────────
    // Uncomment this block and comment out the Phase 1 block above once the
    // bot can reliably score on the ground.  These rewards teach rotation,
    // defense, boost management, and advanced mechanics (dribbles, flicks).
    //
    // std::vector<WeightedReward> rewards = {
    //     // Aggressive goal reward: conceding is penalised 5× harder
    //     { new GoalReward(-5.0f),                                     20.0f },
    //     // Zero-sum ball-to-goal velocity (team-shared, 0.5 spirit)
    //     { new ZeroSumReward(new VelocityBallToGoalReward(), 0.5f),    4.5f },
    //     // Win kickoffs
    //     { new KickoffReward(),                                         1.7f },
    //     // Reward picking up boost so the bot learns boost management
    //     { new PickupBoostReward(),                                     0.3f },
    //     // Shadow defense: stay between ball and own goal
    //     { new ShadowDefenseReward(),                                   0.7f },
    //     // Reward defensive saves
    //     { new SaveReward(),                                            3.5f },
    //     // Strong touch directed toward the opponent goal
    //     { new DirectionalStrongTouchReward(),                          0.7f },
    //     // Encourage correct field rotation (stay behind the ball)
    //     { new FieldRotationReward(),                                   1.0f },
    //     // Dribble: balance ball on roof of car
    //     { new StrictDribbleReward(),                                   1.0f },
    //     // Aerial flick toward goal
    //     { new MawkzyFlickReward(),                                     3.5f },
    // };
    // ──────────────────────────────────────────────────────────────────────

    std::vector<TerminalCondition*> terminalConditions = {
        new NoTouchCondition(30),
        new GoalScoreCondition()
    };

    int playersPerTeam = 1;
    auto arena = Arena::Create(GameMode::SOCCAR);
    for (int i = 0; i < playersPerTeam; i++) {
        arena->AddCar(Team::BLUE);
        arena->AddCar(Team::ORANGE);
    }

    EnvCreateResult result = {};
    result.actionParser = new DefaultAction();
    result.obsBuilder = new DefaultObs();
    result.stateSetter = new RandomStateSetter();
    result.terminalConditions = terminalConditions;
    result.rewards = rewards;
    result.arena = arena;

    return result;
}

// =========================================
// Main
// =========================================
int main(int argc, char* argv[]) {
    RocketSim::Init("collision_meshes");

    LearnerConfig cfg = {};
    cfg.deviceType = LearnerDeviceType::GPU_CUDA;
    cfg.tickSkip = 8;
    cfg.actionDelay = cfg.tickSkip - 1;

    // ── Hardware tuning: RTX 4090 + 30 CPUs + 128 GB RAM ──────────────────
    // 250 parallel arenas saturates 30 CPU cores with headroom for the OS.
    cfg.numGames = 250;
    cfg.randomSeed = 123;

    // Collect 500k steps per iteration (was 300k).  More experience per
    // update improves gradient estimates with the extra CPU throughput.
    int tsPerItr = 500'000;
    cfg.ppo.tsPerItr = tsPerItr;
    cfg.ppo.batchSize = tsPerItr;

    // Larger mini-batch takes advantage of the 4090's 24 GB VRAM.
    cfg.ppo.miniBatchSize = 250'000;

    // One extra epoch squeezes more learning out of each collected batch.
    cfg.ppo.epochs = 3;

    // FP16 (half-precision) for training and inference on the 4090's Tensor
    // Cores – roughly 2× faster GPU throughput with negligible quality loss.
    cfg.ppo.useHalfPrecision = true;

    cfg.ppo.entropyScale = 0.01f;
    cfg.ppo.gaeGamma = 0.99;
    cfg.ppo.policyLR = 2e-4;
    cfg.ppo.criticLR = 2e-4;
    // ──────────────────────────────────────────────────────────────────────

    cfg.ppo.sharedHead.layerSizes = {};
    cfg.ppo.policy.layerSizes = { 1024, 1024, 1024, 1024, 1024, 512 };
    cfg.ppo.critic.layerSizes = { 1024, 1024, 1024, 512 };

    auto optim = ModelOptimType::ADAM;
    cfg.ppo.policy.optimType = optim;
    cfg.ppo.critic.optimType = optim;
    cfg.ppo.sharedHead.optimType = optim;

    auto activation = ModelActivationType::RELU;
    cfg.ppo.policy.activationType = activation;
    cfg.ppo.critic.activationType = activation;
    cfg.ppo.sharedHead.activationType = activation;

    bool addLayerNorm = true;
    cfg.ppo.policy.addLayerNorm = addLayerNorm;
    cfg.ppo.critic.addLayerNorm = addLayerNorm;
    cfg.ppo.sharedHead.addLayerNorm = addLayerNorm;

    cfg.sendMetrics = true;
    cfg.metricsProjectName = "yxllowtechlarge";
    cfg.metricsGroupName = "bot";
    cfg.metricsRunName = "run2";
    cfg.renderMode = false;

    // ── Self-play against old versions (15% chance per iteration) ─────────
    // Every 25M steps a snapshot of the current policy is saved to disk.
    // On ~15% of iterations the bot plays against one of these old snapshots
    // (randomly chosen, on a randomly assigned team) instead of a clone of
    // itself.  This prevents the bot from exploiting its own predictable
    // patterns and encourages more robust play.
    cfg.savePolicyVersions    = true;
    cfg.tsPerVersion          = 25'000'000; // Save a version every 25M steps
    cfg.maxOldVersions        = 32;         // Keep up to 32 old snapshots in memory
    cfg.trainAgainstOldVersions = true;
    cfg.trainAgainstOldChance   = 0.15f;   // 15% of iterations vs an old version
    // ──────────────────────────────────────────────────────────────────────

    // Enable skill tracker to compute and log MMR (ELO-based rating)
    cfg.skillTracker.enabled = true;
    cfg.skillTracker.numArenas = 16;
    cfg.skillTracker.simTime = 45;
    cfg.skillTracker.maxSimTime = 240;
    cfg.skillTracker.updateInterval = 16;
    cfg.skillTracker.ratingInc = 5;

    Learner* learner = new Learner(EnvCreateFunc, cfg);
    learner->Start();

    return EXIT_SUCCESS;
}
