/*
 * AIAudioBridgeControls.cpp - controls for AI audio bridge effect
 */

#include "AIAudioBridgeControls.h"

#include <QByteArray>

#include "AIAudioBridge.h"
#include "Engine.h"
#include "lmms_constants.h"

namespace lmms
{

AIAudioBridgeControls::AIAudioBridgeControls(AIAudioBridgeEffect* effect) :
	EffectControls(effect),
	m_effect(effect),
	m_bridgeEnabledModel(true, this, tr("Bridge enabled")),
	m_mixModel(100.0f, 0.0f, 100.0f, 0.1f, this, tr("AI mix")),
	m_driveModel(1.0f, 0.1f, 4.0f, 0.01f, this, tr("Drive")),
	m_transportModeModel(0, 0, 1, this, tr("Transport mode")),
	m_backendModel(0, 0, 2, this, tr("Backend")),
	m_command("python"),
	m_scriptPath("scripts/ai_audio_bridge_server.py")
{
}


void AIAudioBridgeControls::saveSettings(QDomDocument& doc, QDomElement& parent)
{
	m_bridgeEnabledModel.saveSettings(doc, parent, "bridgeEnabled");
	m_mixModel.saveSettings(doc, parent, "mix");
	m_driveModel.saveSettings(doc, parent, "drive");
	m_transportModeModel.saveSettings(doc, parent, "transportMode");
	m_backendModel.saveSettings(doc, parent, "backend");

	parent.setAttribute("command", m_command);
	parent.setAttribute("scriptPath", m_scriptPath);
	parent.setAttribute("apiKey", m_apiKey);
}


void AIAudioBridgeControls::loadSettings(const QDomElement& parent)
{
	m_bridgeEnabledModel.loadSettings(parent, "bridgeEnabled");
	m_mixModel.loadSettings(parent, "mix");
	m_driveModel.loadSettings(parent, "drive");
	m_transportModeModel.loadSettings(parent, "transportMode");
	m_backendModel.loadSettings(parent, "backend");

	m_command = parent.attribute("command", "python");
	m_scriptPath = parent.attribute("scriptPath", "scripts/ai_audio_bridge_server.py");
	m_apiKey = parent.attribute("apiKey", QString());
}


AIAudioBridgeTransport::Config AIAudioBridgeControls::transportConfig() const
{
	AIAudioBridgeTransport::Config cfg;
	cfg.transportMode = m_transportModeModel.value() == 0
		? AIAudioBridgeTransport::TransportMode::Pipe
		: AIAudioBridgeTransport::TransportMode::SharedMemory;

	switch (m_backendModel.value())
	{
	case 1:
		cfg.backend = AIAudioBridgeTransport::Backend::Gemini;
		break;
	case 2:
		cfg.backend = AIAudioBridgeTransport::Backend::DeepSeek;
		break;
	case 0:
	default:
		cfg.backend = AIAudioBridgeTransport::Backend::Local;
		break;
	}

	cfg.command = m_command.trimmed().isEmpty() ? QString("python") : m_command.trimmed();
	cfg.scriptPath = m_scriptPath.trimmed().isEmpty()
		? QString("scripts/ai_audio_bridge_server.py")
		: m_scriptPath.trimmed();

	if (!m_apiKey.trimmed().isEmpty())
	{
		cfg.apiKey = m_apiKey.trimmed();
	}
	else if (cfg.backend == AIAudioBridgeTransport::Backend::Gemini)
	{
		cfg.apiKey = QString::fromLocal8Bit(qgetenv("GEMINI_API_KEY")).trimmed();
	}
	else if (cfg.backend == AIAudioBridgeTransport::Backend::DeepSeek)
	{
		cfg.apiKey = QString::fromLocal8Bit(qgetenv("DEEPSEEK_API_KEY")).trimmed();
	}

	cfg.sampleRate = Engine::audioEngine()->outputSampleRate();
	cfg.channels = DEFAULT_CHANNELS;
	cfg.timeoutMs = 8;
	cfg.drive = m_driveModel.value();
	return cfg;
}

} // namespace lmms
