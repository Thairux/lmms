/*
 * AIAudioBridgeControlDialog.cpp - UI for AI audio bridge effect
 */

#include "AIAudioBridgeControlDialog.h"

#include <QComboBox>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QVBoxLayout>

#include "AIAudioBridgeControls.h"
#include "Knob.h"

namespace lmms::gui
{

AIAudioBridgeControlDialog::AIAudioBridgeControlDialog(AIAudioBridgeControls* controls) :
	EffectControlDialog(controls),
	m_controls(controls)
{
	setMinimumWidth(430);

	auto* root = new QVBoxLayout(this);

	auto* knobRow = new QHBoxLayout();
	auto* mixKnob = new Knob(KnobType::Bright26, tr("MIX"), this);
	mixKnob->setModel(&m_controls->m_mixModel);
	mixKnob->setHintText(tr("AI mix"), "%");
	mixKnob->setVolumeKnob(true);
	knobRow->addWidget(mixKnob);

	auto* driveKnob = new Knob(KnobType::Bright26, tr("DRIVE"), this);
	driveKnob->setModel(&m_controls->m_driveModel);
	driveKnob->setHintText(tr("Bridge drive"), "x");
	knobRow->addWidget(driveKnob);

	root->addLayout(knobRow);

	auto* form = new QFormLayout();

	m_transportMode = new QComboBox(this);
	m_transportMode->addItem(tr("Pipe (external process)"));
	m_transportMode->addItem(tr("Shared Memory (low latency)"));
	m_transportMode->setCurrentIndex(m_controls->m_transportModeModel.value());
	connect(m_transportMode, qOverload<int>(&QComboBox::currentIndexChanged),
		this, [this](int idx) { m_controls->m_transportModeModel.setValue(idx); });
	form->addRow(tr("Transport"), m_transportMode);

	m_backend = new QComboBox(this);
	m_backend->addItem(tr("Local"));
	m_backend->addItem(tr("Gemini"));
	m_backend->addItem(tr("DeepSeek"));
	m_backend->setCurrentIndex(m_controls->m_backendModel.value());
	connect(m_backend, qOverload<int>(&QComboBox::currentIndexChanged),
		this, [this](int idx) { m_controls->m_backendModel.setValue(idx); });
	form->addRow(tr("Backend"), m_backend);

	m_command = new QLineEdit(m_controls->m_command, this);
	connect(m_command, &QLineEdit::textChanged, this,
		[this](const QString& text) { m_controls->m_command = text; });
	form->addRow(tr("Command"), m_command);

	m_scriptPath = new QLineEdit(m_controls->m_scriptPath, this);
	connect(m_scriptPath, &QLineEdit::textChanged, this,
		[this](const QString& text) { m_controls->m_scriptPath = text; });
	form->addRow(tr("Script"), m_scriptPath);

	m_apiKey = new QLineEdit(m_controls->m_apiKey, this);
	m_apiKey->setEchoMode(QLineEdit::Password);
	connect(m_apiKey, &QLineEdit::textChanged, this,
		[this](const QString& text) { m_controls->m_apiKey = text; });
	form->addRow(tr("API key"), m_apiKey);

	root->addLayout(form);

	auto* note = new QLabel(
		tr("Pipe mode talks to scripts/ai_audio_bridge_server.py.\n"
		   "Shared Memory mode runs local low-latency shaping with no API."),
		this);
	note->setWordWrap(true);
	root->addWidget(note);
}

} // namespace lmms::gui
