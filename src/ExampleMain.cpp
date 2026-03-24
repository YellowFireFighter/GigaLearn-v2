#include <GigaLearnCPP/Learner.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/ObsBuilders/AdvancedObs.h>
#include <RLGymCPP/StateSetters/KickoffState.h>
#include <RLGymCPP/StateSetters/RandomState.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>
#include <RLGymCPP/Rewards/Reward.h>
#include <RLGymCPP/Rewards/testrewards.h>
#include "RewardSchedule.h"
#include <atomic>

using namespace GGL;
using namespace RLGC;

static std::atomic<uint64_t> g_totalSteps{ 0 };

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

class AirTouchReward : public Reward {
public:
    float GetReward(const Player& player, const GameState& state, bool isFinal) override {
        if (!player.ballTouchedStep || player.isOnGround) return 0.f;
        float airFrac = RS_MIN(1.f, player.airTime / 1.75f);
        float heightFrac = state.ball.pos.z / CommonValues::CEILING_Z;
        return RS_MIN(airFrac, heightFrac);
    }
};

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

#pragma once
#include <RLGymCPP/StateSetters/StateSetter.h>
#include <RLGymCPP/StateSetters/KickoffState.h>
#include <RLGymCPP/StateSetters/RandomState.h>
#include <RLGymCPP/CommonValues.h>
#include <cstdlib>
#include <cmath>

using namespace RLGC;

// ── Helpers ───────────────────────────────────────────────
static float RandFloat(float min, float max) {
    return min + (float)rand() / RAND_MAX * (max - min);
}
static float RandSign() {
    return (rand() % 2 == 0) ? 1.f : -1.f;
}

// ── Scenario 1: Ball in air, car on ground looking up ─────
// Teaches the bot to jump and hit aerial balls
class AerialScenarioSetter : public StateSetter {
public:
    void ResetArena(Arena* arena) override {
        arena->ResetToRandomKickoff();

        float ballX = RandFloat(-2000.f, 2000.f);
        float ballY = RandFloat(-3000.f, 3000.f);
        float ballZ = RandFloat(400.f, 1800.f);  // high in the air

        // Give ball some velocity toward goal
        float goalY = CommonValues::BACK_NET_Y;
        Vec ballVel = Vec(
            RandFloat(-200.f, 200.f),
            RandFloat(-300.f, 300.f),
            RandFloat(-100.f, 200.f)
        );

        for (Car* car : arena->GetCars()) {
            if (car->team == Team::BLUE) {
                // Place car under/near ball on ground
                CarState cs = car->GetState();
                cs.pos     = Vec(ballX + RandFloat(-300.f, 300.f),
                                 ballY + RandFloat(-500.f, -200.f), 17.f);
                cs.vel     = Vec(RandFloat(-100.f, 100.f),
                                 RandFloat(200.f, 600.f), 0.f);
                cs.boost   = RandFloat(40.f, 100.f);
                cs.isOnGround = true;
                car->SetState(cs);
            } else {
                // Orange just sits back
                CarState cs = car->GetState();
                cs.pos   = Vec(RandFloat(-500.f, 500.f), 4000.f, 17.f);
                cs.vel   = Vec(0, 0, 0);
                cs.boost = 100.f;
                car->SetState(cs);
            }
        }

        BallState bs = arena->ball->GetState();
        bs.pos = Vec(ballX, ballY, ballZ);
        bs.vel = ballVel;
        bs.angVel = Vec(0, 0, 0);
        arena->ball->SetState(bs);
    }
};

// ── Scenario 2: Car already in air, ball nearby ───────────
// Teaches aerial control and redirects
class MidAirScenarioSetter : public StateSetter {
public:
    void ResetArena(Arena* arena) override {
        arena->ResetToRandomKickoff();

        float carX  = RandFloat(-1500.f, 1500.f);
        float carY  = RandFloat(-2000.f, 2000.f);
        float carZ  = RandFloat(300.f, 1200.f);

        float ballX = carX + RandFloat(-400.f, 400.f);
        float ballY = carY + RandFloat(-400.f, 400.f);
        float ballZ = carZ + RandFloat(-200.f, 400.f);
        ballZ = fmaxf(100.f, fminf(ballZ, 1800.f));

        for (Car* car : arena->GetCars()) {
            if (car->team == Team::BLUE) {
                CarState cs = car->GetState();
                cs.pos  = Vec(carX, carY, carZ);
                cs.vel  = Vec(RandFloat(-300.f, 300.f),
                              RandFloat(200.f, 800.f),
                              RandFloat(-100.f, 300.f));
                cs.boost      = RandFloat(20.f, 80.f);
                cs.isOnGround = false;
                // Random rotation - facing roughly toward ball
                cs.rotMat = RotMat::GetIdentity();
                car->SetState(cs);
            } else {
                CarState cs = car->GetState();
                cs.pos   = Vec(RandFloat(-500.f, 500.f), 4500.f, 17.f);
                cs.vel   = Vec(0, 0, 0);
                cs.boost = 100.f;
                car->SetState(cs);
            }
        }

        BallState bs = arena->ball->GetState();
        bs.pos    = Vec(ballX, ballY, ballZ);
        bs.vel    = Vec(RandFloat(-200.f, 200.f),
                        RandFloat(-200.f, 200.f),
                        RandFloat(-300.f, 0.f));
        bs.angVel = Vec(0, 0, 0);
        arena->ball->SetState(bs);
    }
};

// ── Scenario 3: Flip reset setup ──────────────────────────
// Ball on ceiling or very high, car approaching from below
// Bot needs to jump, flip into ball underside, land on it
class FlipResetScenarioSetter : public StateSetter {
public:
    void ResetArena(Arena* arena) override {
        arena->ResetToRandomKickoff();

        // Ball near ceiling
        float ballX = RandFloat(-1000.f, 1000.f);
        float ballY = RandFloat(-1000.f, 1000.f);
        float ballZ = RandFloat(1600.f, 1900.f); // near ceiling (2044)

        // Car below ball, already moving upward fast
        float carX = ballX + RandFloat(-200.f, 200.f);
        float carY = ballY + RandFloat(-300.f, 300.f);
        float carZ = RandFloat(600.f, 1000.f);

        for (Car* car : arena->GetCars()) {
            if (car->team == Team::BLUE) {
                CarState cs = car->GetState();
                cs.pos  = Vec(carX, carY, carZ);
                cs.vel  = Vec(RandFloat(-100.f, 100.f),
                              RandFloat(-100.f, 100.f),
                              RandFloat(600.f, 1000.f)); // moving up fast
                cs.boost      = RandFloat(50.f, 100.f);
                cs.isOnGround = false;
                car->SetState(cs);
            } else {
                CarState cs = car->GetState();
                cs.pos   = Vec(0.f, 4000.f, 17.f);
                cs.vel   = Vec(0, 0, 0);
                cs.boost = 100.f;
                car->SetState(cs);
            }
        }

        BallState bs = arena->ball->GetState();
        bs.pos    = Vec(ballX, ballY, ballZ);
        bs.vel    = Vec(RandFloat(-100.f, 100.f),
                        RandFloat(-100.f, 100.f),
                        RandFloat(-50.f, 50.f)); // mostly stationary
        bs.angVel = Vec(0, 0, 0);
        arena->ball->SetState(bs);
    }
};

// ── Scenario 4: Wall play ─────────────────────────────────
// Ball rolling down wall, car on wall
class WallPlayScenarioSetter : public StateSetter {
public:
    void ResetArena(Arena* arena) override {
        arena->ResetToRandomKickoff();

        // Pick a wall (left or right)
        float wallX = RandSign() * CommonValues::SIDE_WALL_X;
        float ballY = RandFloat(-2000.f, 2000.f);
        float ballZ = RandFloat(200.f, 1200.f);

        for (Car* car : arena->GetCars()) {
            if (car->team == Team::BLUE) {
                CarState cs = car->GetState();
                // Put car on wall below ball
                cs.pos  = Vec(wallX * 0.95f,
                              ballY + RandFloat(-400.f, 0.f),
                              ballZ - RandFloat(100.f, 400.f));
                cs.vel  = Vec(-wallX * RandFloat(0.f, 200.f), // away from wall
                              RandFloat(0.f, 300.f),
                              RandFloat(100.f, 400.f));
                cs.boost      = RandFloat(30.f, 80.f);
                cs.isOnGround = false;
                car->SetState(cs);
            } else {
                CarState cs = car->GetState();
                cs.pos   = Vec(-wallX * 2000.f, ballY, 17.f);
                cs.vel   = Vec(0, 0, 0);
                cs.boost = 100.f;
                car->SetState(cs);
            }
        }

        BallState bs = arena->ball->GetState();
        bs.pos    = Vec(wallX * 0.9f, ballY, ballZ);
        bs.vel    = Vec(-wallX * RandFloat(100.f, 400.f), // bouncing off wall
                        RandFloat(-200.f, 200.f),
                        RandFloat(-200.f, 100.f));
        bs.angVel = Vec(0, 0, 0);
        arena->ball->SetState(bs);
    }
};

// ── Master setter — randomly picks a scenario ─────────────
class SkillStateSetter : public StateSetter {
public:
    KickoffState           kickoff;
    RandomState            random;
    AerialScenarioSetter   aerial;
    MidAirScenarioSetter   midair;
    FlipResetScenarioSetter flipReset;
    WallPlayScenarioSetter  wallPlay;

    SkillStateSetter() : random(true, true, false) {}

    void ResetArena(Arena* arena) override {
        int roll = rand() % 100;

        if      (roll < 50) kickoff.ResetArena(arena);   // 20% kickoffs
        else if (roll < 80) random.ResetArena(arena);    // 25% random - most important
        else if (roll < 85) aerial.ResetArena(arena);    // 15% aerial
        else if (roll < 90) midair.ResetArena(arena);    // 15% mid-air
        else if (roll < 95) wallPlay.ResetArena(arena);  // 15% wall play
        else                flipReset.ResetArena(arena); // 10% flip reset
    }
};

EnvCreateResult EnvCreateFunc(int index) {
    std::vector<WeightedReward> rewards = {
        { new GoalReward(),                   200.0f },
        { new TouchBallReward(),                3.0f },
        { new VelocityPlayerToBallReward(),     0.5f },
        { new VelocityBallToGoalReward(),       8.0f },
        { new AirReward(),                     0.15f },
        { new TouchAccelReward(),               3.0f },
        //{ new SaveBoostReward(),                0.5f },
        //{ new PickupBoostReward(),              0.3f }
    };

    std::vector<TerminalCondition*> terminalConditions = {
        new NoTouchCondition(80),
        new GoalScoreCondition()
    };

    int playersPerTeam = 2;
    auto arena = Arena::Create(GameMode::SOCCAR);
    for (int i = 0; i < playersPerTeam; i++) {
        arena->AddCar(Team::BLUE);
        arena->AddCar(Team::ORANGE);
    }

    EnvCreateResult result = {};
    result.actionParser = new DefaultAction();
    result.obsBuilder = new AdvancedObs();
    result.stateSetter = new SkillStateSetter();
    result.terminalConditions = terminalConditions;
    result.rewards = rewards;
    result.arena = arena;

    return result;
}

int main(int argc, char* argv[]) {
    RocketSim::Init("collision_meshes");

    LearnerConfig cfg = {};
    cfg.deviceType = LearnerDeviceType::GPU_CUDA;
    cfg.tickSkip = 8;
    cfg.actionDelay = cfg.tickSkip - 1;

    cfg.numGames = 600;
    cfg.randomSeed = 123;

    int tsPerItr = 500'000;
    cfg.ppo.tsPerItr = tsPerItr;
    cfg.ppo.batchSize = tsPerItr;

    cfg.ppo.miniBatchSize = 250'000;

    cfg.ppo.epochs = 2;

    cfg.ppo.entropyScale = 0.025f;
    cfg.ppo.gaeGamma = 0.993;
    cfg.ppo.policyLR = 2e-4;
    cfg.ppo.criticLR = 2e-4;

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
    cfg.metricsRunName = "chronos";
    cfg.renderMode = false;

    cfg.ppo.useHalfPrecision = false;

    cfg.savePolicyVersions    = true;   // Keep old versions so 1.0B milestone can enable self-play
    cfg.tsPerVersion          = 25'000'000;
    cfg.maxOldVersions        = 8;
    cfg.trainAgainstOldVersions = true;  // Enabled automatically at 1.0B by MilestoneTracker
    cfg.trainAgainstOldChance   = 0.15f; 

    cfg.tsPerSave = 100'000'000;

    cfg.skillTracker.enabled = true;
    cfg.skillTracker.numArenas = 16;
    cfg.skillTracker.simTime = 45;
    cfg.skillTracker.maxSimTime = 240;
    cfg.skillTracker.updateInterval = 16;
    cfg.skillTracker.ratingInc = 5;

    bool renderMode = true;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--render") {
            cfg.sendMetrics = false;  
            cfg.ppo.deterministic = true; 
            cfg.renderMode = true;     
            renderMode = true;
            break;
        }
    }

    MilestoneTracker milestoneTracker;

    auto stepCallback = [&milestoneTracker](Learner* learner,
        const std::vector<RLGC::GameState>&,
        Report&) {
            g_totalSteps.store(learner->totalTimesteps);
            milestoneTracker.CheckAndApply(learner);
        };

    Learner* learner = new Learner(EnvCreateFunc, cfg, stepCallback);
    learner->Start();

    return EXIT_SUCCESS;
}
